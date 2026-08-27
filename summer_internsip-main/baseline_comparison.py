"""
TopoTrader V2 -- Baseline Comparison Suite
===========================================
Tests 10 baselines on the same 6 walk-forward windows used by V2.

Baseline Categories:
  1. Naive     : Always-Up, Random, Momentum-1, Momentum-5
  2. Technical : RSI threshold, MACD crossover, MA crossover
  3. Classical ML: Logistic Regression, Random Forest (same 16 features as V2)
  4. DL Ablation : LSTM on OHLCV (5ch), TCN on 7 standard indicators only

Usage:
    python baseline_comparison.py           # US market (default)
    python baseline_comparison.py --market india
    python baseline_comparison.py --quick   # 3 windows, fast mode

Output:
    reports/baseline_comparison_{market}.csv
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ── Project imports ────────────────────────────────────────────────────────────
from topo_trader.utils.data_loader import (
    fetch_universe_tickers, fetch_india_csv_tickers,
    fetch_and_prepare_data, load_india_csv_data, generate_features,
)
from topo_trader.evaluation.walk_forward import (
    WALK_FORWARD_WINDOWS, INDIA_WALK_FORWARD_WINDOWS,
    create_dataset_for_range,
)
from topo_trader.models.tcn import MarketTCN

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_LEN  = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==============================================================================
# HELPER: Common metric computation
# ==============================================================================

def compute_metrics(probs: np.ndarray, y_true: np.ndarray,
                    long_thresh=0.55, short_thresh=0.45) -> dict:
    """
    Compute hit rate on confident predictions.
    probs: (N,) float array in [0, 1]   -- P(up)
    y_true: (N,) binary int array
    """
    if len(probs) == 0:
        return {"overall_acc": 0.0, "hit_rate": 0.0, "confident_pct": 0.0, "n_test": 0}

    preds          = (probs > 0.5).astype(int)
    overall_acc    = accuracy_score(y_true.astype(int), preds)
    confident_mask = (probs > long_thresh) | (probs < short_thresh)

    if confident_mask.sum() > 0:
        hit_rate    = accuracy_score(y_true[confident_mask].astype(int),
                                     preds[confident_mask])
        conf_pct    = confident_mask.mean()
    else:
        hit_rate = overall_acc    # fallback: use overall acc if no confident trades
        conf_pct = 1.0

    return {
        "overall_acc"   : round(overall_acc, 4),
        "hit_rate"      : round(hit_rate, 4),
        "confident_pct" : round(conf_pct, 4),
        "n_test"        : len(probs),
    }


# ==============================================================================
# CATEGORY 1: NAIVE BASELINES
# These operate on raw feature DataFrames, not windowed tensors.
# ==============================================================================

def build_raw_dataset(ticker_features, tickers, start_date, end_date):
    """
    Build a flat (N, n_features) array using just the LAST day of each window.
    Used by classical ML and naive baselines.
    Returns X (N, n_features), y (N,), and the raw log-return series per ticker.
    """
    all_X, all_y, all_returns = [], [], []

    for ticker in tickers:
        if ticker not in ticker_features:
            continue
        df   = ticker_features[ticker]
        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        dfr  = df.loc[mask]

        if len(dfr) < 2 or "C1_LogRet" not in dfr.columns:
            continue

        # Features: current day values (all 16 channels)
        feat_vals = dfr.values[:-1].astype(np.float32)       # (T-1, 16)
        feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)

        # Label: next day direction
        returns   = dfr["C1_LogRet"].values
        labels    = (returns[1:] > 0).astype(int)             # (T-1,)

        all_X.append(feat_vals)
        all_y.append(labels)
        all_returns.append(returns)

    if not all_X:
        return np.empty((0, 16)), np.empty((0,)), np.empty((0,))

    return (np.vstack(all_X),
            np.concatenate(all_y),
            np.concatenate(all_returns))


# ── 1a. Always-Up Baseline ──────────────────────────────────────────────────
def baseline_always_up(X, y):
    probs = np.ones(len(y)) * 0.9   # always confidently predict "up"
    return compute_metrics(probs, y)


# ── 1b. Random Baseline ─────────────────────────────────────────────────────
def baseline_random(X, y):
    rng   = np.random.default_rng(SEED)
    probs = rng.uniform(0, 1, size=len(y))
    return compute_metrics(probs, y)


# ── 1c. Momentum-1 (yesterday's direction) ──────────────────────────────────
def baseline_momentum_1(ticker_features, tickers, test_s, test_e):
    """Uses C1_LogRet sign of today to predict tomorrow."""
    all_probs, all_y = [], []
    for ticker in tickers:
        if ticker not in ticker_features:
            continue
        df   = ticker_features[ticker]
        mask = (df.index >= pd.Timestamp(test_s)) & (df.index <= pd.Timestamp(test_e))
        dfr  = df.loc[mask]
        if len(dfr) < 2 or "C1_LogRet" not in dfr.columns:
            continue
        ret    = dfr["C1_LogRet"].values
        # Today's return sign -> predict tomorrow (shift by 1)
        signal = (ret[:-1] > 0).astype(float)   # 1=up, 0=down
        probs  = np.where(signal == 1, 0.65, 0.35)   # map to confident probabilities
        labels = (ret[1:] > 0).astype(int)
        all_probs.append(probs)
        all_y.append(labels)
    if not all_probs:
        return {"overall_acc": 0.0, "hit_rate": 0.0, "confident_pct": 0.0, "n_test": 0}
    return compute_metrics(np.concatenate(all_probs), np.concatenate(all_y))


# ── 1d. Momentum-5 (5-day return sign) ──────────────────────────────────────
def baseline_momentum_5(ticker_features, tickers, test_s, test_e):
    """5-day rolling return sign predicts next day."""
    all_probs, all_y = [], []
    for ticker in tickers:
        if ticker not in ticker_features:
            continue
        df   = ticker_features[ticker]
        mask = (df.index >= pd.Timestamp(test_s)) & (df.index <= pd.Timestamp(test_e))
        dfr  = df.loc[mask]
        if len(dfr) < 7 or "C1_LogRet" not in dfr.columns:
            continue
        ret      = dfr["C1_LogRet"].values
        roll5    = pd.Series(ret).rolling(5).sum().values
        signal   = (roll5[:-1] > 0).astype(float)
        probs    = np.where(signal == 1, 0.62, 0.38)
        labels   = (ret[1:] > 0).astype(int)
        valid    = ~np.isnan(roll5[:-1])
        all_probs.append(probs[valid])
        all_y.append(labels[valid])
    if not all_probs:
        return {"overall_acc": 0.0, "hit_rate": 0.0, "confident_pct": 0.0, "n_test": 0}
    return compute_metrics(np.concatenate(all_probs), np.concatenate(all_y))


# ==============================================================================
# CATEGORY 2: TECHNICAL ANALYSIS BASELINES
# ==============================================================================

# ── 2a. RSI Threshold ───────────────────────────────────────────────────────
def baseline_rsi(ticker_features, tickers, test_s, test_e):
    """
    RSI < 35 -> predict up (oversold)
    RSI > 65 -> predict down (overbought)
    else     -> no confident signal (prob = 0.5)
    """
    all_probs, all_y = [], []
    for ticker in tickers:
        if ticker not in ticker_features:
            continue
        df   = ticker_features[ticker]
        mask = (df.index >= pd.Timestamp(test_s)) & (df.index <= pd.Timestamp(test_e))
        dfr  = df.loc[mask]
        if len(dfr) < 2 or "C3_RSI" not in dfr.columns:
            continue
        rsi    = dfr["C3_RSI"].values[:-1]
        ret    = dfr["C1_LogRet"].values
        labels = (ret[1:] > 0).astype(int)
        probs  = np.where(rsi < 0.35, 0.70,         # oversold -> predict up
                 np.where(rsi > 0.65, 0.30, 0.50))  # overbought -> predict down
        all_probs.append(probs)
        all_y.append(labels)
    if not all_probs:
        return {"overall_acc": 0.0, "hit_rate": 0.0, "confident_pct": 0.0, "n_test": 0}
    return compute_metrics(np.concatenate(all_probs), np.concatenate(all_y))


# ── 2b. MACD Crossover ──────────────────────────────────────────────────────
def baseline_macd(ticker_features, tickers, test_s, test_e):
    """MACD > 0 -> predict up, MACD < 0 -> predict down."""
    all_probs, all_y = [], []
    for ticker in tickers:
        if ticker not in ticker_features:
            continue
        df   = ticker_features[ticker]
        mask = (df.index >= pd.Timestamp(test_s)) & (df.index <= pd.Timestamp(test_e))
        dfr  = df.loc[mask]
        if len(dfr) < 2 or "C4_MACD" not in dfr.columns:
            continue
        macd   = dfr["C4_MACD"].values[:-1]
        ret    = dfr["C1_LogRet"].values
        labels = (ret[1:] > 0).astype(int)
        probs  = np.where(macd > 0, 0.65, 0.35)
        all_probs.append(probs)
        all_y.append(labels)
    if not all_probs:
        return {"overall_acc": 0.0, "hit_rate": 0.0, "confident_pct": 0.0, "n_test": 0}
    return compute_metrics(np.concatenate(all_probs), np.concatenate(all_y))


# ── 2c. Bollinger %B Mean Reversion ─────────────────────────────────────────
def baseline_bollinger(ticker_features, tickers, test_s, test_e):
    """%B < 0.2 -> oversold -> up, %B > 0.8 -> overbought -> down."""
    all_probs, all_y = [], []
    for ticker in tickers:
        if ticker not in ticker_features:
            continue
        df   = ticker_features[ticker]
        mask = (df.index >= pd.Timestamp(test_s)) & (df.index <= pd.Timestamp(test_e))
        dfr  = df.loc[mask]
        if len(dfr) < 2 or "C6_BB" not in dfr.columns:
            continue
        bb     = dfr["C6_BB"].values[:-1]
        ret    = dfr["C1_LogRet"].values
        labels = (ret[1:] > 0).astype(int)
        probs  = np.where(bb < 0.20, 0.68,
                 np.where(bb > 0.80, 0.32, 0.50))
        all_probs.append(probs)
        all_y.append(labels)
    if not all_probs:
        return {"overall_acc": 0.0, "hit_rate": 0.0, "confident_pct": 0.0, "n_test": 0}
    return compute_metrics(np.concatenate(all_probs), np.concatenate(all_y))


# ==============================================================================
# CATEGORY 3: CLASSICAL ML BASELINES (on same 16-channel V2 features)
# ==============================================================================

def baseline_logistic_regression(X_train, y_train, X_test, y_test):
    """Logistic Regression on flattened 16-feature snapshot (last day only)."""
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    model = LogisticRegression(C=0.1, max_iter=500, random_state=SEED, n_jobs=-1)
    model.fit(X_tr_sc, y_train.astype(int))

    probs = model.predict_proba(X_te_sc)[:, 1]
    return compute_metrics(probs, y_test)


def baseline_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest on 16-feature snapshot."""
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6,
        min_samples_leaf=50,     # prevents tiny leaves -> overconfident probabilities
        min_samples_split=100,   # prevents splits on very small groups
        random_state=SEED, n_jobs=-1
    )
    model.fit(X_train, y_train.astype(int))
    probs = model.predict_proba(X_test)[:, 1]
    return compute_metrics(probs, y_test)


# ==============================================================================
# CATEGORY 4: DEEP LEARNING ABLATIONS
# ==============================================================================

# ── LSTM on raw OHLCV (5 channels) ──────────────────────────────────────────
class OHLCVDataset:
    """Extracts just [C1_LogRet, C2_Vol, C5_ATR, C6_BB, C7_ZScore] (5ch)."""
    OHLCV_CHANNELS = [0, 1, 4, 5, 6]   # indices in the 16-channel feature array


class SimpleLSTM(nn.Module):
    """2-layer LSTM for directional prediction from a sequence of features."""
    def __init__(self, input_size=5, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout)
        self.head   = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        # x: (batch, features, seq_len) -> permute to (batch, seq_len, features)
        x         = x.permute(0, 2, 1)
        out, _    = self.lstm(x)
        return self.head(out[:, -1, :])   # last time step


def train_lstm(X_train, y_train, n_channels, epochs=30, batch_size=64, lr=1e-3):
    """Generic LSTM trainer — cosine LR, grad clip, early stopping (mirrors train_model)."""
    model = SimpleLSTM(input_size=n_channels, hidden_size=32, num_layers=2, dropout=0.2)
    model.to(DEVICE)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    opt       = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/100)
    crit      = nn.BCELoss()
    N         = len(X_train)
    best_loss, wait, patience, min_delta = float('inf'), 0, 5, 1e-5

    model.train()
    for epoch in range(epochs):
        perm, epoch_loss, nb = torch.randperm(N), 0.0, 0
        for i in range(0, N, batch_size):
            idx    = perm[i:i+batch_size]
            bx, by = X_t[idx], y_t[idx]
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item(); nb += 1
        scheduler.step()
        mean_loss = epoch_loss / max(nb, 1)
        if mean_loss < best_loss - min_delta:
            best_loss, wait = mean_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def eval_lstm(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        probs = model(X_t).flatten().cpu().numpy()
    return compute_metrics(probs, y_test)


# ── TCN on 7-channel standard indicators only ───────────────────────────────
def train_tcn_7ch(X_train, y_train, epochs=30, batch_size=64, lr=1e-3):
    """Train MarketTCN (7ch) — cosine LR, grad clip, early stopping (mirrors train_model)."""
    X_7   = X_train[:, :7, :]   # C1–C7 already Z-scored by create_dataset_for_range
    model = MarketTCN(num_inputs=7, num_channels=[32, 32, 32, 32],
                      kernel_size=3, dropout=0.2)
    model.to(DEVICE)
    X_t = torch.tensor(X_7,     dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    opt       = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/100)
    crit      = nn.BCELoss()
    N         = len(X_7)
    best_loss, wait, patience, min_delta = float('inf'), 0, 5, 1e-5

    model.train()
    for epoch in range(epochs):
        perm, nb = torch.randperm(N), 0
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx    = perm[i:i+batch_size]
            bx, by = X_t[idx], y_t[idx]
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item(); nb += 1
        scheduler.step()
        mean_loss = epoch_loss / max(nb, 1)
        if mean_loss < best_loss - min_delta:
            best_loss, wait = mean_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def eval_tcn_7ch(model, X_test, y_test):
    X_7 = X_test[:, :7, :]
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X_7, dtype=torch.float32).to(DEVICE)
        probs = model(X_t).flatten().cpu().numpy()
    return compute_metrics(probs, y_test)


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_baseline_comparison(market="us", quick=False):
    os.makedirs("reports", exist_ok=True)
    windows = INDIA_WALK_FORWARD_WINDOWS if market == "india" else WALK_FORWARD_WINDOWS
    if quick:
        windows = windows[:3]

    # ── 1. Load data & features ────────────────────────────────────────────
    print(f"\n=== Baseline Comparison -- {market.upper()} Market ===")
    print(f"Running {len(windows)} walk-forward windows on {DEVICE}\n")

    if market == "india":
        tickers = fetch_india_csv_tickers()
        data    = load_india_csv_data(start_date="2015-01-01", end_date="2022-12-31")
        tickers = list(data.columns.get_level_values(0).unique())
    else:
        tickers = fetch_universe_tickers()
        data    = fetch_and_prepare_data(tickers, start_date="2015-01-01",
                                         end_date="2024-01-01")

    print("Generating V2 features (16 channels)...")
    features, _ = generate_features(data, tickers, parallel=False, n_jobs=1)
    print(f"Features ready for {len(features)} tickers.\n")

    # ── 2. Define all baselines ────────────────────────────────────────────
    all_results = []

    for i, (train_s, train_e, test_s, test_e, label) in enumerate(windows):
        print(f"\n{'='*65}")
        print(f"Window {i+1}/{len(windows)} -- {label}")
        print(f"  Train: {train_s} -> {train_e}   |   Test: {test_s} -> {test_e}")

        # ── Build datasets ──────────────────────────────────────────────
        # Windowed tensors for TCN-based models
        X_tr_16, y_tr, _tr_stats = create_dataset_for_range(features, tickers, WINDOW_LEN, train_s, train_e)
        X_te_16, y_te, _          = create_dataset_for_range(features, tickers, WINDOW_LEN, test_s,  test_e,
                                                              channel_stats=_tr_stats)

        if len(X_tr_16) == 0 or len(X_te_16) == 0:
            print("  SKIP -- insufficient data")
            continue

        # Flat snapshots for classical ML (last day of each window = current values)
        X_tr_flat, y_tr_flat, _ = build_raw_dataset(features, tickers, train_s, train_e)
        X_te_flat, y_te_flat, _ = build_raw_dataset(features, tickers, test_s,  test_e)

        # OHLCV-only windowed tensors (channels 0-4)
        OHLCV_IDX = [0, 1, 4, 5, 6]
        X_tr_ohlcv = X_tr_16[:, OHLCV_IDX, :]
        X_te_ohlcv = X_te_16[:, OHLCV_IDX, :]

        window_rows = []

        def add_result(name, category, metrics):
            row = {
                "window"       : i + 1,
                "regime"       : label,
                "test_period"  : f"{test_s} -> {test_e}",
                "method"       : name,
                "category"     : category,
                "overall_acc"  : metrics["overall_acc"],
                "hit_rate"     : metrics["hit_rate"],
                "confident_pct": metrics["confident_pct"],
                "n_test"       : metrics["n_test"],
            }
            all_results.append(row)
            print(f"  [{name:35s}]  hit={metrics['hit_rate']:.3f}  "
                  f"conf={metrics['confident_pct']:.1%}  n={metrics['n_test']:,}", flush=True)

        # ── Category 1: Naive ───────────────────────────────────────────
        add_result("Always-Up",       "Naive", baseline_always_up(X_te_flat, y_te_flat))
        add_result("Random",          "Naive", baseline_random(X_te_flat, y_te_flat))
        add_result("Momentum-1",      "Naive", baseline_momentum_1(features, tickers, test_s, test_e))
        add_result("Momentum-5",      "Naive", baseline_momentum_5(features, tickers, test_s, test_e))

        # ── Category 2: Technical ───────────────────────────────────────
        add_result("RSI Threshold",   "Technical", baseline_rsi(features, tickers, test_s, test_e))
        add_result("MACD Crossover",  "Technical", baseline_macd(features, tickers, test_s, test_e))
        add_result("Bollinger %B",    "Technical", baseline_bollinger(features, tickers, test_s, test_e))

        # ── Category 3: Classical ML ────────────────────────────────────
        if len(X_tr_flat) > 0 and len(X_te_flat) > 0:
            add_result("Logistic Regression (16ch)", "Classical ML",
                       baseline_logistic_regression(X_tr_flat, y_tr_flat, X_te_flat, y_te_flat))
            add_result("Random Forest (16ch)",       "Classical ML",
                       baseline_random_forest(X_tr_flat, y_tr_flat, X_te_flat, y_te_flat))

        # ── Category 4: DL Ablations ────────────────────────────────────
        print(f"  Training LSTM-OHLCV (5ch, 30 epochs)...", flush=True)
        lstm_ohlcv = train_lstm(X_tr_ohlcv, y_tr, n_channels=5, epochs=30)
        add_result("LSTM - OHLCV only (5ch)",  "DL Ablation", eval_lstm(lstm_ohlcv, X_te_ohlcv, y_te))

        print(f"  Training TCN-7ch (std indicators, 10 epochs)...", flush=True)
        tcn_7 = train_tcn_7ch(X_tr_16, y_tr, epochs=10)
        add_result("TCN - Std Indicators (7ch)", "DL Ablation", eval_tcn_7ch(tcn_7, X_te_16, y_te))

    # ── 3. Save & print results ────────────────────────────────────────────
    if not all_results:
        print("\nNo results generated. Check data availability.")
        return

    df = pd.DataFrame(all_results)
    out_path = f"reports/baseline_comparison_{market}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n\nResults saved -> {out_path}")

    # ── 4. Summary table ───────────────────────────────────────────────────
    print("\n" + "="*75)
    print("SUMMARY: Mean Hit Rate Across All Windows")
    print("="*75)
    print(f"{'Method':<38} {'Category':<15} {'Mean Hit Rate':>13} {'Mean Conf%':>11}")
    print("-"*75)

    summary = (df.groupby(["method", "category"])[["hit_rate", "confident_pct"]]
                 .mean()
                 .sort_values("hit_rate", ascending=False))

    for (method, category), row in summary.iterrows():
        marker = " <-- YOUR MODEL" if "TopoTrader" in method else ""
        print(f"  {method:<36} {category:<15} {row['hit_rate']:>12.1%} "
              f"{row['confident_pct']:>10.1%}{marker}")

    print("="*75)
    print("\nNOTE: TopoTrader V2 results are in reports/walk_forward_{market}.csv")
    print("      Merge that file with this one for the complete comparison table.")

    return df


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TopoTrader V2 Baseline Comparison")
    parser.add_argument("--market", default="us", choices=["us", "india"],
                        help="Market to run (default: us)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 windows, 8 epochs (for testing)")
    args = parser.parse_args()

    results_df = run_baseline_comparison(market=args.market, quick=args.quick)
