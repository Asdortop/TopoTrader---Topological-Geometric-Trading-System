"""
ablation_study.py
=================
TopoTrader V2 — Ablation Study + Statistical Inference
India Market (Nifty-50), 8 walk-forward OOS windows (2014–2021)

Ablation Variants:
  1. V2-Full      : All 16 channels (baseline to beat)
  2. V2-NoRegime  : Zero out C13-C16 (regime labels removed)
  3. V2-NoGAT     : Replace C8 GAT with V1 Laplacian (use_gat=False)
  4. V2-NoTDA     : Zero out C10_H0, C11_H1 (topology removed)
  5. V2-NoWalsh   : Zero out C9_Walsh (WHT removed)

Statistical Inference Per Ablation:
  - Per-window binomial test: hit_rate > 0.5 (one-sided)
  - Across-window paired t-test: V2-Full vs ablation (window obs)
  - Wilson 95% CI on mean hit rate
  - Cohen's d effect size

Run:
    python ablation_study.py --market india
"""

import os
import sys
import argparse
import traceback
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest

sys.stdout.reconfigure(line_buffering=True)

from topo_trader.utils.data_loader import (
    fetch_india_csv_tickers, load_india_csv_data,
    generate_features, create_dataset,
)
from topo_trader.evaluation.walk_forward import (
    INDIA_WALK_FORWARD_WINDOWS, create_dataset_for_range,
)
from topo_trader.train import train_model

# ── Config ─────────────────────────────────────────────────────────────────────
WINDOW_LEN  = 64
EPOCHS      = 10
BATCH_SIZE  = 64
LR          = 1e-3
CONF_THRESH = 0.55

ABLATION_ZERO = {
    "V2-NoRegime" : ["C13_Regime_Crash", "C14_Regime_HighVol",
                     "C15_Regime_Bull",  "C16_Regime_Sideways"],
    "V2-NoTDA"    : ["C10_H0", "C11_H1"],
    "V2-NoWalsh"  : ["C9_Walsh"],
}


# ── Statistical Helpers ────────────────────────────────────────────────────────

def wilson_ci(hits, n, z=1.96):
    if n == 0:
        return 0.5, 0.5
    p      = hits / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return round(centre - margin, 4), round(centre + margin, 4)


def binomial_pvalue(hits, n, p0=0.5):
    if n == 0:
        return 1.0
    result = binomtest(int(hits), int(n), p0, alternative='greater')
    return result.pvalue


def paired_ttest(rates_a, rates_b):
    if len(rates_a) < 2:
        return float('nan'), float('nan')
    diffs  = [a - b for a, b in zip(rates_a, rates_b)]
    t, p2  = stats.ttest_1samp(diffs, 0)
    p1     = p2 / 2 if t > 0 else 1 - p2 / 2
    return round(t, 3), round(p1, 4)


def cohens_d(rates_a, rates_b):
    diffs = [a - b for a, b in zip(rates_a, rates_b)]
    if len(diffs) < 2 or np.std(diffs) == 0:
        return 0.0
    return round(np.mean(diffs) / np.std(diffs, ddof=1), 3)


# ── Zero-Channel Mask ─────────────────────────────────────────────────────────

def apply_mask(features: dict, zero_cols: list) -> dict:
    masked = {}
    for ticker, df in features.items():
        overrides = {c: 0.0 for c in zero_cols if c in df.columns}
        masked[ticker] = df.assign(**overrides) if overrides else df
    return masked


# ── Single-Window Evaluation ──────────────────────────────────────────────────

def eval_window(features, tickers, train_s, train_e, test_s, test_e):
    """
    Train Logistic Regression on [train_s, train_e], evaluate on [test_s, test_e].
    Uses LR instead of TCN for ablation: LR cannot collapse to constant predictions
    and directly measures feature contribution without architectural confounds.
    Returns dict: {hit_rate, n_confident, n_total, hits, conf_pct}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import warnings

    try:
        X_tr, y_tr, _tr_stats = create_dataset_for_range(
            features, tickers, WINDOW_LEN, train_s, train_e
        )
        X_te, y_te, _         = create_dataset_for_range(
            features, tickers, WINDOW_LEN, test_s, test_e,
            channel_stats=_tr_stats
        )
    except Exception as e:
        print(f"SKIP(dataset:{e})", flush=True)
        return None

    if len(X_tr) == 0 or len(X_te) == 0:
        print("SKIP(empty)", flush=True)
        return None

    # Flatten (N, C, T) → (N, C*T) for LR
    n_tr, C, T = X_tr.shape
    n_te       = X_te.shape[0]
    X_tr_flat  = X_tr.reshape(n_tr, C * T)
    X_te_flat  = X_te.reshape(n_te, C * T)

    # Scale
    scaler    = StandardScaler()
    X_tr_flat = scaler.fit_transform(X_tr_flat)
    X_te_flat = scaler.transform(X_te_flat)

    # Train LR (C=0.1 for regularisation — prevents overfit on short windows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=0.1, max_iter=500,
                                 solver="lbfgs", random_state=42)
        clf.fit(X_tr_flat, y_tr)

    # Get calibrated probabilities
    probs  = clf.predict_proba(X_te_flat)[:, 1]

    mask   = (probs > CONF_THRESH) | (probs < (1 - CONF_THRESH))
    n_conf = int(mask.sum())
    if n_conf == 0:
        return {"hit_rate": 0.0, "n_confident": 0,
                "n_total": len(y_te), "hits": 0, "conf_pct": 0.0}

    pred  = (probs[mask] > 0.5).astype(int)
    hits  = int((pred == y_te[mask]).sum())
    return {
        "hit_rate"   : round(hits / n_conf, 4),
        "n_confident": n_conf,
        "n_total"    : len(y_te),
        "hits"       : hits,
        "conf_pct"   : round(100 * n_conf / len(y_te), 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ablation(market="india"):
    os.makedirs("reports", exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  TopoTrader V2 — Ablation Study ({market.upper()})")
    print(f"{'='*70}\n")

    # 1. Load data
    print("Loading data...", flush=True)
    tickers = fetch_india_csv_tickers()
    data    = load_india_csv_data(
        start_date="2010-01-01",
        end_date="2021-04-30",
        cache_name="india_csv_data.parquet",
    )
    tickers = list(data.columns.get_level_values(0).unique())
    print(f"Loaded: {len(tickers)} tickers\n", flush=True)

    # 2. Generate V2 features (GAT, 16ch)
    print("Generating V2 features (16ch, GAT)...", flush=True)
    features_full, _ = generate_features(data, tickers, parallel=False, version="v2")

    # 3. Generate V1 features (Laplacian, 12ch) for NoGAT ablation
    print("\nGenerating V1 features (12ch, Laplacian) for NoGAT ablation...", flush=True)
    features_v1, _ = generate_features(data, tickers, parallel=False, version="v1")
    # Pad V1 to 16ch by zeroing regime cols (fair comparison — same input size)
    for ticker in features_v1:
        df = features_v1[ticker]
        for col in ["C13_Regime_Crash", "C14_Regime_HighVol",
                    "C15_Regime_Bull",  "C16_Regime_Sideways"]:
            if col not in df.columns:
                df[col] = 0.0

    # 4. Build ablation dict
    ablations = {
        "V2-Full"     : features_full,
        "V2-NoRegime" : apply_mask(features_full, ABLATION_ZERO["V2-NoRegime"]),
        "V2-NoGAT"    : features_v1,
        "V2-NoTDA"    : apply_mask(features_full, ABLATION_ZERO["V2-NoTDA"]),
        "V2-NoWalsh"  : apply_mask(features_full, ABLATION_ZERO["V2-NoWalsh"]),
    }

    windows     = INDIA_WALK_FORWARD_WINDOWS
    all_results = {name: [] for name in ablations}

    # 5. Walk-forward evaluation
    for wi, (tr_s, tr_e, te_s, te_e, label) in enumerate(windows):
        print(f"\n{'─'*70}")
        print(f"  Window {wi+1}/{len(windows)} — {label}")
        print(f"  Train: {tr_s} → {tr_e}   |   Test: {te_s} → {te_e}")
        print(f"{'─'*70}")

        for abl_name, feats in ablations.items():
            print(f"  [{abl_name:15s}]  ", end="", flush=True)
            result = eval_window(feats, tickers, tr_s, tr_e, te_s, te_e)
            all_results[abl_name].append(result)

            if result is None:
                print("SKIP", flush=True)
            else:
                hr   = result['hit_rate']
                conf = result['conf_pct']
                n    = result['n_confident']
                p    = binomial_pvalue(result['hits'], n)
                star = ("***" if p < 0.001 else
                        "** " if p < 0.01  else
                        "*  " if p < 0.05  else "   ")
                print(f"hit={hr:.3f}  conf={conf:5.1f}%  n={n:,}  "
                      f"p_binom={p:.4f}{star}", flush=True)

    # 6. Aggregate summary
    print(f"\n\n{'='*70}")
    print("  ABLATION SUMMARY — Pooled & Paired Statistics")
    print(f"{'='*70}\n")

    full_valid = [r for r in all_results["V2-Full"] if r is not None]
    full_rates = [r['hit_rate'] for r in full_valid]

    summary_rows = []
    for abl_name, results in all_results.items():
        valid = [r for r in results if r is not None]
        if not valid:
            continue

        rates     = [r['hit_rate']    for r in valid]
        confs     = [r['conf_pct']    for r in valid]
        tot_hits  = sum(r['hits']        for r in valid)
        tot_n     = sum(r['n_confident'] for r in valid)
        mean_hr   = np.mean(rates)
        mean_conf = np.mean(confs)

        ci_lo, ci_hi = wilson_ci(tot_hits, tot_n)
        p_vs50       = binomial_pvalue(tot_hits, tot_n, p0=0.5)

        if abl_name == "V2-Full":
            delta, t_stat, p_pair, d = 0.0, float('nan'), float('nan'), float('nan')
        else:
            matched_full = [full_rates[i] for i, r in enumerate(results)
                            if r is not None and i < len(full_rates)]
            matched_abl  = rates
            delta        = round((np.mean(matched_full) - mean_hr) * 100, 2)
            t_stat, p_pair = paired_ttest(matched_full, matched_abl)
            d            = cohens_d(matched_full, matched_abl)

        summary_rows.append({
            "Ablation"   : abl_name,
            "N_windows"  : len(valid),
            "Mean_Hit%"  : round(mean_hr * 100, 2),
            "CI_95"      : f"[{ci_lo*100:.1f},{ci_hi*100:.1f}]",
            "Mean_Conf%" : round(mean_conf, 1),
            "p_vs_50%"   : round(p_vs50, 5),
            "Delta_vs_Full%" : delta if abl_name != "V2-Full" else "-",
            "p_paired"   : round(p_pair, 4) if not np.isnan(p_pair) else "-",
            "Cohens_d"   : d if not np.isnan(d) else "-",
        })

    df_sum = pd.DataFrame(summary_rows).sort_values("Mean_Hit%", ascending=False)

    # Pretty print
    print(df_sum.to_string(index=False))
    print(f"\nLegend:")
    print(f"  Delta_vs_Full% = hit-rate points LOST when component removed")
    print(f"  p_paired       = one-sided paired t-test (H1: Full > ablation)")
    print(f"  Cohens_d       = effect size (0.2=small, 0.5=medium, 0.8=large)")
    print(f"  p_vs_50%       = binomial test (H1: hit_rate > 50%)")

    # 7. Save
    df_sum.to_csv(f"reports/ablation_study_{market}.csv", index=False)
    print(f"\nResults saved → reports/ablation_study_{market}.csv")

    # Per-window detail
    detail = []
    for wi, (_, _, te_s, _, label) in enumerate(windows):
        for abl_name, results in all_results.items():
            r = results[wi] if wi < len(results) else None
            detail.append({
                "window": wi+1, "label": label, "test_year": te_s[:4],
                "ablation": abl_name,
                "hit_rate": r['hit_rate']    if r else None,
                "conf_pct": r['conf_pct']    if r else None,
                "n_conf"  : r['n_confident'] if r else None,
                "hits"    : r['hits']        if r else None,
                "n_total" : r['n_total']     if r else None,
            })
    pd.DataFrame(detail).to_csv(f"reports/ablation_detail_{market}.csv", index=False)
    print(f"Per-window detail → reports/ablation_detail_{market}.csv")

    return df_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TopoTrader V2 Ablation Study")
    parser.add_argument("--market", default="india", choices=["india", "us"])
    args = parser.parse_args()

    try:
        run_ablation(market=args.market)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except Exception as e:
        print(f"\nFATAL: {e}", flush=True)
        traceback.print_exc()
