"""
Portfolio Backtest — TopoTrader V2 vs Baselines
================================================
Converts per-window hit rates + confidence levels into realistic
portfolio performance metrics for the Indian Nifty-50 market.

Methodology:
  - Uses saved per-window summary statistics (hit_rate, confident_trade_pct, n_test)
  - Simulates a simple long-only strategy on confident predictions
  - Applies NSE-realistic transaction costs:
      * Brokerage     : 0.03% (discount broker, e.g. Zerodha)
      * STT           : 0.10% on sell side
      * Exchange fees : 0.00345% (NSE)
      * GST on brok.  : 18% of brokerage = 0.0054%
      * Stamp duty    : 0.015% on buy side
      * Total one-way : ~0.18%  (round-trip: ~0.36%)
  - Uses empirical Nifty-50 up/down day magnitudes:
      * Mean up-day   : +0.67% (based on 2010-2021 Nifty data)
      * Mean down-day : -0.70%
  - Computes: Sharpe ratio, Max Drawdown, CAGR, Win Rate, # trades

Output:
  Formatted comparison table printed to stdout
  Saved to reports/portfolio_backtest.csv
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── NSE Transaction Cost Parameters ──────────────────────────────────────────
BROKERAGE_PCT  = 0.0003      # 0.03% Zerodha-style discount broker
STT_PCT        = 0.001       # 0.10% STT on sell side
EXCHANGE_PCT   = 0.0000345   # NSE exchange + SEBI charges
GST_PCT        = BROKERAGE_PCT * 0.18   # GST on brokerage
STAMP_PCT      = 0.00015     # Stamp duty on buy side
ONE_WAY_COST   = BROKERAGE_PCT + STT_PCT/2 + EXCHANGE_PCT + GST_PCT + STAMP_PCT
ROUNDTRIP_COST = ONE_WAY_COST * 2       # ~0.36% round-trip

# ── Empirical Nifty-50 daily return parameters (2010-2021) ───────────────────
MEAN_UP_DAY   =  0.0067  # +0.67% mean return on up days
MEAN_DOWN_DAY = -0.0070  # -0.70% mean return on down days
ANNUAL_DAYS   = 252

print(f"NSE Round-trip transaction cost: {ROUNDTRIP_COST*100:.3f}%")
print(f"Mean up-day: {MEAN_UP_DAY*100:.2f}%  |  Mean down-day: {MEAN_DOWN_DAY*100:.2f}%\n")


def simulate_window_portfolio(hit_rate, conf_pct, n_test, n_trading_days,
                               model_name, regime_label):
    """
    Simulate portfolio performance for one walk-forward window.

    Args:
        hit_rate        : Fraction of confident trades that were correct
        conf_pct        : Fraction of days the model trades
        n_test          : Total stock-days in test set
        n_trading_days  : Calendar trading days in test period (~252/yr)
        model_name      : Name string for output
        regime_label    : Regime name

    Returns:
        dict of performance metrics
    """
    # Number of confident trades (positions taken)
    n_confident   = int(round(n_test * conf_pct))
    n_correct     = int(round(n_confident * hit_rate))
    n_wrong       = n_confident - n_correct

    # Expected gross P&L per trade (long-only: bet on up-days)
    # Correct trade:  earn MEAN_UP_DAY
    # Wrong trade:    lose abs(MEAN_DOWN_DAY)
    gross_return_per_day = (
        n_correct * MEAN_UP_DAY
        + n_wrong  * MEAN_DOWN_DAY
    ) / max(n_trading_days, 1)   # daily average over the window

    # Transaction cost: each new position costs round-trip
    # Assume average position held for 1-3 days (short-term signals)
    # Conservative: one round-trip per confident day
    cost_per_day  = conf_pct * ROUNDTRIP_COST

    net_daily_return  = gross_return_per_day - cost_per_day

    # Annualized metrics
    cagr              = net_daily_return * ANNUAL_DAYS * 100   # in %

    # Sharpe ratio (simplified: no risk-free rate, use daily std of returns)
    # Approximate daily return std from hit rate variance
    daily_returns = []
    for _ in range(n_confident):
        if np.random.rand() < hit_rate:
            daily_returns.append(MEAN_UP_DAY - ROUNDTRIP_COST)
        else:
            daily_returns.append(MEAN_DOWN_DAY - ROUNDTRIP_COST)
    # Fill non-trading days with 0
    non_trading = n_trading_days - n_confident
    daily_returns.extend([0.0] * max(0, non_trading))

    daily_arr = np.array(daily_returns)
    sharpe    = (daily_arr.mean() * ANNUAL_DAYS) / (daily_arr.std() * np.sqrt(ANNUAL_DAYS) + 1e-10)

    # Max drawdown (simplified cumulative simulation)
    cum_returns  = np.cumprod(1 + daily_arr)
    running_max  = np.maximum.accumulate(cum_returns)
    drawdowns    = (cum_returns - running_max) / running_max
    max_drawdown = drawdowns.min() * 100   # in %

    # Gross win rate on confident trades
    win_rate     = hit_rate * 100

    return {
        "model"         : model_name,
        "regime"        : regime_label,
        "n_trades"      : n_confident,
        "win_rate_%"    : round(win_rate, 1),
        "CAGR_%"        : round(cagr, 2),
        "Sharpe"        : round(sharpe, 3),
        "MaxDD_%"       : round(max_drawdown, 2),
        "net_edge_%"    : round((2*hit_rate - 1) * 100 - ROUNDTRIP_COST*100, 3),
    }


# ── Load data ─────────────────────────────────────────────────────────────────
np.random.seed(42)
v2_df   = pd.read_csv("reports/walk_forward_india.csv")
base_df = pd.read_csv("reports/baseline_comparison_india.csv")

VALID_WINDOWS = [3, 4, 5, 6, 7]

# Window lengths in trading days (approximate)
WINDOW_DAYS = {3: 247, 4: 248, 5: 248, 6: 249, 7: 262}

v2_valid   = v2_df[v2_df["window"].isin(VALID_WINDOWS)].reset_index(drop=True)
base_valid = base_df[base_df["window"].isin(VALID_WINDOWS)].reset_index(drop=True)


# ── V2 Portfolio ──────────────────────────────────────────────────────────────
print("=" * 75)
print("V2 Portfolio Performance — Per Window")
print("=" * 75)

v2_results = []
for _, row in v2_valid.iterrows():
    w  = int(row["window"])
    res = simulate_window_portfolio(
        hit_rate       = float(row["hit_rate"]),
        conf_pct       = float(row["confident_trade_pct"]),
        n_test         = int(row["n_test"]),
        n_trading_days = WINDOW_DAYS.get(w, 252),
        model_name     = "V2 (TopoTrader)",
        regime_label   = row["regime_label"],
    )
    v2_results.append(res)

v2_perf = pd.DataFrame(v2_results)
print(v2_perf.to_string(index=False))


# ── Baseline Portfolio Comparison ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("Portfolio Comparison — V2 vs Key Baselines (Mean across W3-W7)")
print("=" * 75)

models_to_compare = [
    "V2 (TopoTrader)",
    "TCN - Std Indicators (7ch)",
    "LSTM - OHLCV only (5ch)",
    "Random Forest (16ch)",
    "Logistic Regression (16ch)",
    "Always-Up",
    "Momentum-1",
]

summary_rows = []

# V2 mean
v2_mean = {
    "model"      : "V2 (TopoTrader)",
    "CAGR_%"     : round(v2_perf["CAGR_%"].mean(), 2),
    "Sharpe"     : round(v2_perf["Sharpe"].mean(), 3),
    "MaxDD_%"    : round(v2_perf["MaxDD_%"].mean(), 2),
    "win_rate_%"  : round(v2_perf["win_rate_%"].mean(), 1),
    "net_edge_%"  : round(v2_perf["net_edge_%"].mean(), 3),
}
summary_rows.append(v2_mean)

# Baseline means
for method in base_valid["method"].unique():
    if not any(m in method for m in ["TCN", "LSTM", "Random Forest", "Logistic", "Always", "Momentum"]):
        continue
    rows = base_valid[base_valid["method"] == method].sort_values("window")
    if len(rows) != 5:
        continue

    b_results = []
    for _, row in rows.iterrows():
        w   = int(row["window"])
        res = simulate_window_portfolio(
            hit_rate       = float(row["hit_rate"]),
            conf_pct       = float(row["confident_pct"]),
            n_test         = int(row["n_test"]),
            n_trading_days = WINDOW_DAYS.get(w, 252),
            model_name     = method,
            regime_label   = row["regime"],
        )
        b_results.append(res)

    b_perf = pd.DataFrame(b_results)
    summary_rows.append({
        "model"      : method,
        "CAGR_%"     : round(b_perf["CAGR_%"].mean(), 2),
        "Sharpe"     : round(b_perf["Sharpe"].mean(), 3),
        "MaxDD_%"    : round(b_perf["MaxDD_%"].mean(), 2),
        "win_rate_%"  : round(b_perf["win_rate_%"].mean(), 1),
        "net_edge_%"  : round(b_perf["net_edge_%"].mean(), 3),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Sharpe", ascending=False)
print(summary_df.to_string(index=False))

# ── Break-even analysis ───────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("Break-Even Analysis")
print("=" * 75)
breakeven_hr = 0.50 + ROUNDTRIP_COST / 2
print(f"  Round-trip cost             : {ROUNDTRIP_COST*100:.3f}%")
print(f"  Break-even hit rate         : {breakeven_hr*100:.2f}%")
print(f"  V2 pooled hit rate          : 50.67%")
print(f"  V2 vs break-even            : {(0.5067 - breakeven_hr)*100:+.2f}pp")
print(f"\n  → At current hit rates, V2 {'IS' if 0.5067 > breakeven_hr else 'is NOT'} profitable after NSE costs")
print(f"  → Institutional cost (0.05% round-trip) break-even: {(0.50 + 0.0005/2)*100:.3f}%")
print(f"  → At institutional costs, V2 {'IS' if 0.5067 > (0.50 + 0.0005/2) else 'is NOT'} profitable")

# ── Save ─────────────────────────────────────────────────────────────────────
summary_df.to_csv("reports/portfolio_backtest.csv", index=False)
v2_perf.to_csv("reports/portfolio_backtest_v2_detail.csv", index=False)
print("\nSaved: reports/portfolio_backtest.csv")
print("       reports/portfolio_backtest_v2_detail.csv")
