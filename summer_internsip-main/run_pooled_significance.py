"""
TopoTrader V3 vs V2 — Pooled Significance Analysis
====================================================
Instead of comparing window-level averages (n=5, very low power),
this script pools ALL test predictions across all windows and runs:
  - Binomial test: is pooled hit rate > 50.10% (NSE break-even)?
  - McNemar's test: is V3 correct where V2 isn't, and vice versa?
  - Wilson 95% CI on pooled hit rates

This is the CORRECT statistical test for this setup.
n_pooled ~ 45,000 samples gives proper statistical power.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topo_trader.utils.data_loader import (
    fetch_india_csv_tickers, load_india_csv_data, generate_features,
)
from topo_trader.train import train_model, train_model_v3
from topo_trader.evaluation.walk_forward import (
    INDIA_WALK_FORWARD_WINDOWS, create_dataset_for_range,
)
from topo_trader.models.tcn_v3 import MarketTCN_V3

WINDOW_LEN   = 64
EPOCHS       = 30
LONG_THRESH  = 0.55   # use tight deadband for this test
SHORT_THRESH = 0.45
BREAK_EVEN   = 0.5010  # NSE break-even after costs

def load_features():
    tickers = fetch_india_csv_tickers()
    data = load_india_csv_data(start_date="2010-01-01", end_date="2021-04-30",
                               cache_name="india_csv_data.parquet")
    tickers = list(data.columns.get_level_values(0).unique())
    features, _ = generate_features(data, tickers, parallel=False, n_jobs=1)
    return features, tickers


def get_predictions(model, X_test, device='cpu'):
    model.eval()
    model.to(device)
    with torch.no_grad():
        probs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        probs = probs.flatten().cpu().numpy()
    return probs


def run_pooled_analysis():
    print("Loading features ...", flush=True)
    features, tickers = load_features()

    # Use windows W3-W7 (skip W1-W2 with small training sets)
    windows = INDIA_WALK_FORWARD_WINDOWS[2:7]

    all_probs_v3 = []
    all_probs_v2 = []
    all_labels   = []
    window_results = []

    for i, (train_s, train_e, test_s, test_e, label) in enumerate(windows):
        print(f"\nWindow {i+1}/{len(windows)} — {label}", flush=True)

        X_train, y_train, stats = create_dataset_for_range(
            features, tickers, WINDOW_LEN, train_s, train_e)
        X_test,  y_test,  _    = create_dataset_for_range(
            features, tickers, WINDOW_LEN, test_s, test_e, channel_stats=stats)

        if len(X_train) == 0 or len(X_test) == 0:
            print("  SKIP"); continue

        print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}", flush=True)

        # Train V3
        model_v3 = train_model_v3(X_train, y_train, epochs=EPOCHS, lr=1e-3)
        # Train V2
        model_v2 = train_model(X_train, y_train, epochs=EPOCHS, lr=1e-3)

        probs_v3 = get_predictions(model_v3, X_test)
        probs_v2 = get_predictions(model_v2, X_test)

        all_probs_v3.append(probs_v3)
        all_probs_v2.append(probs_v2)
        all_labels.append(y_test)

        # Per-window summary
        conf_mask_v3 = (probs_v3 > LONG_THRESH) | (probs_v3 < SHORT_THRESH)
        conf_mask_v2 = (probs_v2 > LONG_THRESH) | (probs_v2 < SHORT_THRESH)
        hr_v3 = (probs_v3[conf_mask_v3] > 0.5).astype(int) == y_test[conf_mask_v3].astype(int)
        hr_v2 = (probs_v2[conf_mask_v2] > 0.5).astype(int) == y_test[conf_mask_v2].astype(int)
        window_results.append({
            "window": label,
            "v3_hr": hr_v3.mean() if len(hr_v3) > 0 else 0,
            "v2_hr": hr_v2.mean() if len(hr_v2) > 0 else 0,
            "v3_cov": conf_mask_v3.mean(),
            "v2_cov": conf_mask_v2.mean(),
            "n_test": len(X_test),
        })

    # ── Pool all predictions ──────────────────────────────────────────────────
    probs_v3 = np.concatenate(all_probs_v3)
    probs_v2 = np.concatenate(all_probs_v2)
    labels   = np.concatenate(all_labels)

    print(f"\n{'='*65}")
    print(f"POOLED ANALYSIS  (n = {len(labels):,} total test predictions)")
    print(f"{'='*65}")

    for name, probs in [("V3", probs_v3), ("V2", probs_v2)]:
        conf_mask = (probs > LONG_THRESH) | (probs < SHORT_THRESH)
        preds     = (probs > 0.5).astype(int)
        hr        = (preds[conf_mask] == labels[conf_mask].astype(int)).mean()
        n_conf    = conf_mask.sum()
        ci_lo, ci_hi = proportion_confint(int(hr * n_conf), n_conf,
                                          alpha=0.05, method='wilson')
        binom     = sp.binomtest(int(hr * n_conf), n_conf, p=BREAK_EVEN,
                                 alternative='greater')
        print(f"\n  {name}:")
        print(f"    Pooled Hit Rate   : {hr:.4f}  ({hr*100:.2f}%)")
        print(f"    Confident trades  : {n_conf:,}  ({conf_mask.mean():.1%} of total)")
        print(f"    Wilson 95% CI     : [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"    vs break-even {BREAK_EVEN:.4f}: p = {binom.pvalue:.4f}")
        if binom.pvalue < 0.05:
            print(f"    => STATISTICALLY SIGNIFICANT above break-even!")
        elif ci_lo > 0.50:
            print(f"    => Lower CI bound > 0.50: practically significant")

    # McNemar's test: V3 vs V2 on same predictions
    conf_both = ((probs_v3 > LONG_THRESH) | (probs_v3 < SHORT_THRESH)) & \
                ((probs_v2 > LONG_THRESH) | (probs_v2 < SHORT_THRESH))
    preds_v3  = (probs_v3[conf_both] > 0.5).astype(int)
    preds_v2  = (probs_v2[conf_both] > 0.5).astype(int)
    y_both    = labels[conf_both].astype(int)

    v3_right_v2_wrong = ((preds_v3 == y_both) & (preds_v2 != y_both)).sum()
    v3_wrong_v2_right = ((preds_v3 != y_both) & (preds_v2 == y_both)).sum()
    mcnemar_stat = (abs(v3_right_v2_wrong - v3_wrong_v2_right) - 1) ** 2 / \
                    (v3_right_v2_wrong + v3_wrong_v2_right + 1e-8)
    mcnemar_p = sp.chi2.sf(mcnemar_stat, df=1)

    print(f"\n  McNemar's Test (V3 vs V2 on overlapping confident trades):")
    print(f"    V3 right, V2 wrong  : {v3_right_v2_wrong:,}")
    print(f"    V3 wrong, V2 right  : {v3_wrong_v2_right:,}")
    print(f"    chi2 = {mcnemar_stat:.3f},  p = {mcnemar_p:.4f}")
    if mcnemar_p < 0.05:
        if v3_right_v2_wrong > v3_wrong_v2_right:
            print(f"    => V3 IS SIGNIFICANTLY BETTER than V2 (p < 0.05)")
        else:
            print(f"    => V2 IS SIGNIFICANTLY BETTER than V3 (p < 0.05)")
    else:
        print(f"    => V3 and V2 are statistically equivalent (p = {mcnemar_p:.4f})")

    # Per-window summary table
    print(f"\n{'='*65}")
    print("PER-WINDOW SUMMARY")
    print(f"{'Window':<28} {'V3 HR':>8} {'V2 HR':>8} {'Gap':>8} {'V3 Cov':>8}")
    print(f"{'-'*65}")
    for r in window_results:
        gap = (r['v3_hr'] - r['v2_hr']) * 100
        print(f"  {r['window']:<26} {r['v3_hr']:>8.4f} {r['v2_hr']:>8.4f} "
              f"{gap:>+7.2f}pp {r['v3_cov']:>7.1%}")

    # Save
    os.makedirs("reports", exist_ok=True)
    pd.DataFrame(window_results).to_csv("reports/pooled_significance_v3.csv", index=False)
    print("\nSaved -> reports/pooled_significance_v3.csv")


if __name__ == "__main__":
    run_pooled_analysis()
