"""
TopoTrader V3 — Walk-Forward Training & Evaluation Runner
===========================================================
Runs V3 (Dual-Branch TCN + SE-Net + MoE) on India Nifty-50 walk-forward windows
and compares against V2 baseline under identical conditions.

Usage
-----
    python run_training_v3.py

Output
------
    reports/walk_forward_v3_india.csv    — V3 per-window results
    reports/v3_vs_v2_comparison.csv      — side-by-side comparison
    topo_trader/models/tcn_v3_wX.pth     — saved model per window
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Add parent to path so topo_trader imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topo_trader.train import train_model, train_model_v3, save_model
from topo_trader.evaluation.walk_forward import (
    INDIA_WALK_FORWARD_WINDOWS,
    create_dataset_for_range,
    evaluate_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_LEN      = 64        # time steps per sample
EPOCHS          = 30
BATCH_SIZE      = 64
LR              = 1e-3
LONG_THRESH     = 0.53      # V3 output distribution is tighter — widened from 0.55
SHORT_THRESH    = 0.47      # widened from 0.45 to capture more confident trades
SAVE_MODELS     = True
REPORT_DIR      = "reports"
MODEL_DIR       = "topo_trader/models"
USE_INDIA       = True

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load features
# ─────────────────────────────────────────────────────────────────────────────

def load_features():
    """Load India feature data using the same pipeline as run_training_v2.py."""
    from topo_trader.utils.data_loader import (
        fetch_india_csv_tickers,
        load_india_csv_data,
        generate_features,
    )
    tickers = fetch_india_csv_tickers()
    print(f"Found {len(tickers)} India tickers.", flush=True)

    data = load_india_csv_data(
        start_date="2010-01-01",
        end_date  ="2021-04-30",
        cache_name="india_csv_data.parquet",
    )
    tickers = list(data.columns.get_level_values(0).unique())
    print(f"Generating 16-channel features for {len(tickers)} tickers ...", flush=True)

    ticker_features, _ = generate_features(data, tickers, parallel=False, n_jobs=1)
    print(f"Feature shape: {next(iter(ticker_features.values())).shape}", flush=True)
    return ticker_features, tickers


def get_magnitudes(ticker_features, tickers, start_date, end_date, window_len):
    """
    Extract per-sample absolute return magnitude for magnitude-weighted BCE.
    Aligns exactly with create_dataset_for_range sample order.
    """
    magnitudes = []
    for ticker in tickers:
        if ticker not in ticker_features:
            continue
        df   = ticker_features[ticker]
        mask = (df.index >= pd.Timestamp(start_date)) & \
               (df.index <= pd.Timestamp(end_date))
        df_r = df.loc[mask]
        if len(df_r) < window_len + 1 or "C1_LogRet" not in df_r.columns:
            continue
        log_rets   = df_r["C1_LogRet"].values
        n_samples  = len(log_rets) - window_len - 1
        if n_samples <= 0:
            continue
        # Magnitude for sample i = abs return at position i + window_len
        for i in range(n_samples):
            magnitudes.append(abs(float(log_rets[i + window_len])))
    return np.array(magnitudes, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_v3_walk_forward(ticker_features, tickers, windows):
    results_v3 = []
    results_v2 = []

    for i, (train_s, train_e, test_s, test_e, label) in enumerate(windows):
        print(f"\n{'='*65}")
        print(f"Window {i+1}/{len(windows)} — {label}")
        print(f"  Train: {train_s} -> {train_e}  |  Test: {test_s} -> {test_e}")

        # ── Build dataset ─────────────────────────────────────────────────
        X_train, y_train, stats = create_dataset_for_range(
            ticker_features, tickers, WINDOW_LEN, train_s, train_e
        )
        X_test, y_test, _ = create_dataset_for_range(
            ticker_features, tickers, WINDOW_LEN, test_s, test_e,
            channel_stats=stats
        )

        if len(X_train) == 0 or len(X_test) == 0:
            print("  SKIP — insufficient data"); continue

        print(f"  Train: {len(X_train):,} samples  |  Test: {len(X_test):,} samples")

        # ── Magnitudes for weighted loss ──────────────────────────────────
        mags_train = get_magnitudes(ticker_features, tickers,
                                    train_s, train_e, WINDOW_LEN)
        # Align lengths (magnitudes may differ slightly due to NaN skipping)
        min_len    = min(len(X_train), len(mags_train))
        X_train_   = X_train[:min_len]
        y_train_   = y_train[:min_len]
        mags_train_= mags_train[:min_len]

        # ── Train V3 ──────────────────────────────────────────────────────
        print(f"\n  [V3] Training ...", flush=True)
        model_v3 = train_model_v3(
            X_train_, y_train_,
            magnitudes_train = mags_train_,
            epochs     = EPOCHS,
            batch_size = BATCH_SIZE,
            lr         = LR,
        )

        if SAVE_MODELS:
            save_model(model_v3,
                f"{MODEL_DIR}/tcn_v3_w{i+1}_{label.replace('/', '_').replace(' ', '_')}.pth")

        # ── Train V2 (fair comparison — same epochs, same LR) ─────────────
        print(f"\n  [V2] Training ...", flush=True)
        model_v2 = train_model(
            X_train, y_train,
            epochs     = EPOCHS,
            batch_size = BATCH_SIZE,
            lr         = LR,
        )

        # ── Evaluate both ─────────────────────────────────────────────────
        m_v3 = evaluate_model(model_v3, X_test, y_test, LONG_THRESH, SHORT_THRESH)
        m_v2 = evaluate_model(model_v2, X_test, y_test, LONG_THRESH, SHORT_THRESH)

        gap  = (m_v3["hit_rate"] - m_v2["hit_rate"]) * 100

        print(f"\n  {'─'*50}")
        print(f"  {'Metric':<25} {'V3':>10} {'V2':>10} {'Gap':>10}")
        print(f"  {'─'*50}")
        print(f"  {'Hit Rate':<25} {m_v3['hit_rate']:>10.4f} {m_v2['hit_rate']:>10.4f} {gap:>+9.2f}pp")
        print(f"  {'Overall Accuracy':<25} {m_v3['overall_accuracy']:>10.4f} {m_v2['overall_accuracy']:>10.4f}")
        print(f"  {'Confident Trade %':<25} {m_v3['confident_trade_pct']:>10.1%} {m_v2['confident_trade_pct']:>10.1%}")

        results_v3.append({"window": i+1, "label": label, "model": "V3", **m_v3})
        results_v2.append({"window": i+1, "label": label, "model": "V2", **m_v2})

    return pd.DataFrame(results_v3), pd.DataFrame(results_v2)


# ─────────────────────────────────────────────────────────────────────────────
# Summary & Significance
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df_v3, df_v2):
    from scipy import stats as sp

    print(f"\n{'='*65}")
    print("WALK-FORWARD SUMMARY — V3 vs V2")
    print(f"{'='*65}")
    print(f"\n{'Window':<6} {'Regime':<25} {'V3 HR':>8} {'V2 HR':>8} {'Gap':>8}")
    print(f"{'─'*60}")

    for _, (r3, r2) in enumerate(zip(df_v3.itertuples(), df_v2.itertuples())):
        gap = (r3.hit_rate - r2.hit_rate) * 100
        flag = "✓" if gap > 0 else " "
        print(f"  W{r3.window}   {r3.label:<23} {r3.hit_rate:>8.4f} {r2.hit_rate:>8.4f} {gap:>+7.2f}pp {flag}")

    print(f"{'─'*60}")
    print(f"  {'MEAN':<28} {df_v3['hit_rate'].mean():>8.4f} {df_v2['hit_rate'].mean():>8.4f}")

    # Paired t-test
    if len(df_v3) >= 3:
        t_stat, p_val = sp.ttest_rel(df_v3["hit_rate"], df_v2["hit_rate"])
        mean_diff = (df_v3["hit_rate"] - df_v2["hit_rate"]).mean() * 100
        print(f"\n  Paired t-test (V3 vs V2): t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Mean V3 advantage: {mean_diff:+.3f} pp")
        if p_val < 0.05:
            print(f"  ✅ V3 SIGNIFICANTLY BETTER than V2 (p < 0.05)")
        elif p_val < 0.20:
            print(f"  🟡 V3 trending better but not yet significant (p={p_val:.3f})")
        else:
            print(f"  ⬜ V3 at parity with V2 (p={p_val:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("TopoTrader V3 — Walk-Forward Training")
    print("=" * 65)

    ticker_features, tickers = load_features()

    # Use W3-W7 (skip W1/W2 with insufficient data, skip W8 with 4 months)
    active_windows = INDIA_WALK_FORWARD_WINDOWS[2:7]

    df_v3, df_v2 = run_v3_walk_forward(ticker_features, tickers, active_windows)

    print_summary(df_v3, df_v2)

    # Save results
    combined = pd.concat([df_v3, df_v2], ignore_index=True)
    out_path = f"{REPORT_DIR}/walk_forward_v3_india.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")
