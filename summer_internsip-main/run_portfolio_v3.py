"""
TopoTrader V3 — Multi-Stock Portfolio Simulation (Week 9)
===========================================================
Runs a full walk-forward portfolio backtest across all 49 Nifty stocks.

For each walk-forward window:
  1. Trains V3 model on historical data
  2. Runs inference on test period, all 49 stocks daily
  3. Allocates portfolio using Kelly sizing + top-K selection
  4. Applies stop-losses and circuit breakers
  5. Records equity curve, Sharpe, MaxDD, CAGR

Usage:
    python run_portfolio_v3.py

Output:
    reports/portfolio_equity_curve.csv
    reports/portfolio_summary.csv
    Console: formatted metrics table
"""

import os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topo_trader.models.tcn_v3 import MarketTCN_V3
from topo_trader.train import train_model_v3
from topo_trader.evaluation.walk_forward import (
    INDIA_WALK_FORWARD_WINDOWS, create_dataset_for_range,
)
from topo_trader.utils.data_loader import (
    fetch_india_csv_tickers, load_india_csv_data, generate_features,
)
from topo_trader.strategies.risk_manager import RiskManager
from topo_trader.strategies.portfolio_allocator import PortfolioAllocator, PortfolioBacktester

os.makedirs("reports", exist_ok=True)

WINDOW_LEN    = 64
EPOCHS        = 30
TOP_K         = 7
INITIAL_CAP   = 1_000_000   # 10 lakh INR
ANNUAL_DAYS   = 252
RISK_FREE     = 0.065       # India 10-yr G-Sec


# ─────────────────────────────────────────────────────────────────────────────

def load_features():
    tickers = fetch_india_csv_tickers()
    data = load_india_csv_data(start_date="2010-01-01", end_date="2021-04-30",
                               cache_name="india_csv_data.parquet")
    tickers = list(data.columns.get_level_values(0).unique())
    features, _ = generate_features(data, tickers, parallel=False, n_jobs=1)
    return features, tickers, data


def get_daily_returns(data: pd.DataFrame, tickers: list, start: str, end: str) -> pd.DataFrame:
    """Extract daily log returns for all tickers between two dates."""
    rows = {}
    for t in tickers:
        if t not in data.columns.get_level_values(0):
            continue
        try:
            close = data[t]['Close']
            mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))
            c = close.loc[mask]
            if len(c) > 1:
                rows[t] = np.log(c / c.shift(1)).iloc[1:]
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_prob_df(model: MarketTCN_V3, features: dict, tickers: list,
                  start: str, end: str, channel_stats=None) -> pd.DataFrame:
    """
    Build a (dates x tickers) DataFrame of daily model probabilities.
    Applies the same C1-C7 Z-score normalisation that create_dataset_for_range uses,
    so the model sees the same input distribution as during training.

    channel_stats: (ch_mean, ch_std) — shape (7,) arrays from training window.
    """
    model.eval()
    model.cpu()
    prob_rows = {}
    N_PRICE_CH = 7

    # Unpack normalisation stats
    if channel_stats is not None:
        ch_mean, ch_std = channel_stats
        safe_std = np.where(ch_std < 1e-8, 1.0, ch_std)
    else:
        ch_mean, ch_std, safe_std = None, None, None

    for t in tickers:
        if t not in features:
            continue
        df = features[t]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        dr = df.loc[mask]
        if len(dr) < WINDOW_LEN + 1:
            continue

        raw = dr.values.astype(np.float32)           # (T, C)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply C1-C7 normalisation (same as create_dataset_for_range)
        if ch_mean is not None:
            raw[:, :N_PRICE_CH] = (
                (raw[:, :N_PRICE_CH] - ch_mean[np.newaxis, :]) / safe_std[np.newaxis, :]
            )
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        vals = raw.T   # (16, T)
        dates_out = []
        probs_out = []

        with torch.no_grad():
            for i in range(WINDOW_LEN, len(dr)):
                window = torch.tensor(
                    vals[:, i-WINDOW_LEN:i], dtype=torch.float32
                ).unsqueeze(0)   # (1, 16, 64)
                p = float(model(window).flatten()[0])
                dates_out.append(dr.index[i])
                probs_out.append(p)

        prob_rows[t] = pd.Series(probs_out, index=dates_out)

    if not prob_rows:
        return pd.DataFrame()
    return pd.DataFrame(prob_rows)


def get_regime_series(features: dict, tickers: list, start: str, end: str) -> pd.Series:
    """Extract regime labels from C13-C16 channels."""
    for t in tickers:
        if t not in features:
            continue
        df = features[t]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        dr = df.loc[mask]
        # Regime one-hot: columns C13-C16
        regime_cols = [c for c in dr.columns if c.startswith('C13') or
                       c.startswith('C14') or c.startswith('C15') or c.startswith('C16')]
        if len(regime_cols) >= 4:
            regime_vals = dr[regime_cols].values
            regime_idx  = np.argmax(regime_vals, axis=1)
            return pd.Series(regime_idx.tolist(), index=dr.index, name='regime')
    return pd.Series(dtype=int)


# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("Loading features ...", flush=True)
    features, tickers, raw_data = load_features()

    windows = INDIA_WALK_FORWARD_WINDOWS[2:7]   # W3-W7 for richer training

    allocator = PortfolioAllocator(top_k=TOP_K, kelly_fraction=1.0, allow_short=True)
    backtester = PortfolioBacktester(allocator=allocator)

    all_equity = []
    window_summaries = []

    for i, (train_s, train_e, test_s, test_e, label) in enumerate(windows):
        print(f"\n[Window {i+1}/5] {label}  (test: {test_s} -> {test_e})", flush=True)

        # ── Train V3 ─────────────────────────────────────────────────────────
        X_tr, y_tr, stats = create_dataset_for_range(features, tickers, WINDOW_LEN,
                                                      train_s, train_e)
        if len(X_tr) == 0:
            print("  Skipping — no training data."); continue

        print(f"  Training V3 on {len(X_tr):,} samples ...", flush=True)
        model = train_model_v3(X_tr, y_tr, epochs=EPOCHS, lr=1e-3)
        model.eval()

        # ── Build probability grid for test period ────────────────────────────
        print("  Building probability grid ...", flush=True)
        prob_df   = build_prob_df(model, features, tickers, test_s, test_e, stats)
        ret_df    = get_daily_returns(raw_data, tickers, test_s, test_e)
        regime_sr = get_regime_series(features, tickers, test_s, test_e)

        if prob_df.empty or ret_df.empty:
            print("  Skipping — empty prob/return data."); continue

        # Align dates
        common_dates = prob_df.index.intersection(ret_df.index)
        prob_df   = prob_df.loc[common_dates]
        ret_df    = ret_df.loc[common_dates]
        regime_sr = regime_sr.reindex(common_dates, fill_value=3)

        print(f"  Running portfolio simulation: {len(common_dates)} trading days ...", flush=True)
        results = backtester.run(prob_df, ret_df, regime_sr,
                                 initial_capital=INITIAL_CAP)

        backtester.print_summary(results)

        # Record
        eq = results['equity_curve']
        eq.name = label
        all_equity.append(eq)

        window_summaries.append({
            'window':         label,
            'sharpe':         round(results['sharpe'], 3),
            'max_drawdown':   round(results['max_drawdown'], 4),
            'cagr':           round(results['cagr'], 4),
            'win_rate':       round(results['win_rate'], 4),
            'n_trades':       results['n_trades'],
            'final_capital':  round(results['final_capital'], 0),
            **{f"sharpe_{k}": round(v, 3)
               for k, v in results['per_regime_sharpe'].items()},
        })

    # ── Overall summary ───────────────────────────────────────────────────────
    if not window_summaries:
        print("No results to summarise."); return

    df = pd.DataFrame(window_summaries)
    print("\n" + "=" * 70)
    print("OVERALL PORTFOLIO SUMMARY — V3 Walk-Forward")
    print("=" * 70)
    print(f"{'Window':<25} {'Sharpe':>7} {'MaxDD':>7} {'CAGR':>7} {'WinRate':>8} {'Trades':>7}")
    print("-" * 70)
    for _, r in df.iterrows():
        sharpe_flag = " *" if r['sharpe'] > 0.5 else ""
        print(f"  {r['window']:<23} {r['sharpe']:>7.3f} "
              f"{r['max_drawdown']:>7.2%} {r['cagr']:>7.2%} "
              f"{r['win_rate']:>8.2%} {r['n_trades']:>7,}{sharpe_flag}")

    print("-" * 70)
    print(f"  {'MEAN':<23} {df['sharpe'].mean():>7.3f} "
          f"{df['max_drawdown'].mean():>7.2%} {df['cagr'].mean():>7.2%} "
          f"{df['win_rate'].mean():>8.2%}")
    print("\n  * = Sharpe > 0.5 (target achieved)")

    # Save
    df.to_csv("reports/portfolio_summary.csv", index=False)
    if all_equity:
        # Drop duplicate dates within each series before concat
        clean_eq = [e[~e.index.duplicated(keep='first')] for e in all_equity]
        equity_df = pd.concat(clean_eq, axis=1, join='outer')
        equity_df.to_csv("reports/portfolio_equity_curve.csv")
        print("\nSaved:")
        print("  reports/portfolio_summary.csv")
        print("  reports/portfolio_equity_curve.csv")


if __name__ == "__main__":
    run()
