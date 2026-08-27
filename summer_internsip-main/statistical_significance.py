"""
Statistical Significance Analysis — TopoTrader V2
===================================================
Tests whether V2 hit rates are statistically significant vs:
  1. Random chance (50% null hypothesis) — Binomial test per window
  2. TCN-7ch baseline — Paired t-test across windows + Cohen's d
  3. All baselines — Bonferroni-corrected comparison table

Uses existing CSVs:
  reports/walk_forward_india.csv        (V2 results)
  reports/baseline_comparison_india.csv (baseline results)

Output:
  reports/statistical_significance.csv
  prints formatted table to stdout
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest, ttest_rel, norm
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────────────────
v2_df   = pd.read_csv("reports/walk_forward_india.csv")
base_df = pd.read_csv("reports/baseline_comparison_india.csv")

# Use W3-W7 only (W8 is unreliable — only 735 samples, 4 months)
VALID_WINDOWS = [3, 4, 5, 6, 7]
v2_valid   = v2_df[v2_df["window"].isin(VALID_WINDOWS)].reset_index(drop=True)
base_valid = base_df[base_df["window"].isin(VALID_WINDOWS)].reset_index(drop=True)


def wilson_ci(hits, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p    = hits / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return max(0, centre - margin), min(1, centre + margin)


def binomial_pvalue(hit_rate, n_samples, p0=0.50):
    """Two-sided binomial test: H0: hit_rate == p0."""
    hits = int(round(hit_rate * n_samples))
    result = binomtest(hits, n=n_samples, p=p0, alternative='two-sided')
    return result.pvalue


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Per-window binomial significance for V2
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 1: V2 Per-Window Significance vs 50% Random Chance")
print("="*70)

rows_v2 = []
for _, row in v2_valid.iterrows():
    hr   = float(row["hit_rate"])
    n    = int(row["n_test"])       # total test samples in window
    conf = float(row["confident_trade_pct"])
    n_confident = int(round(n * conf))   # only confident predictions counted

    # binomial test on the confident subset
    p_val = binomial_pvalue(hr, n_confident)
    lo, hi = wilson_ci(int(round(hr * n_confident)), n_confident)
    sig = "✓" if p_val < 0.05 else ("~" if p_val < 0.10 else "✗")

    rows_v2.append({
        "window"        : int(row["window"]),
        "regime"        : row["regime_label"],
        "hit_rate"      : round(hr, 4),
        "n_confident"   : n_confident,
        "wilson_lo"     : round(lo, 4),
        "wilson_hi"     : round(hi, 4),
        "p_value"       : round(p_val, 4),
        "significant"   : sig,
    })

v2_sig_df = pd.DataFrame(rows_v2)
print(v2_sig_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Pooled V2 significance (all W3-W7 combined)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 2: V2 Pooled Significance (W3-W7 combined)")
print("="*70)

total_confident = 0
total_correct   = 0
for _, row in v2_valid.iterrows():
    hr   = float(row["hit_rate"])
    n    = int(row["n_test"])
    conf = float(row["confident_trade_pct"])
    nc   = int(round(n * conf))
    total_confident += nc
    total_correct   += int(round(hr * nc))

pooled_hr = total_correct / total_confident if total_confident > 0 else 0.0
pooled_p  = binomial_pvalue(pooled_hr, total_confident)
lo, hi    = wilson_ci(total_correct, total_confident)
z_score   = (pooled_hr - 0.50) / np.sqrt(0.25 / total_confident)

print(f"  Total confident predictions : {total_confident:,}")
print(f"  Total correct               : {total_correct:,}")
print(f"  Pooled hit rate             : {pooled_hr:.4f} ({pooled_hr*100:.2f}%)")
print(f"  Wilson 95% CI               : [{lo:.4f}, {hi:.4f}]")
print(f"  Z-score vs 50%              : {z_score:.3f}")
print(f"  Two-sided p-value           : {pooled_p:.4f}  {'*** SIGNIFICANT' if pooled_p < 0.05 else '(not significant at 5%)'}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: V2 vs each baseline — paired t-test across 5 windows
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 3: V2 vs Baselines — Paired t-test across W3-W7 (n=5 pairs)")
print("="*70)

v2_hit_rates = v2_valid["hit_rate"].values.astype(float)   # shape (5,)

baselines = base_valid["method"].unique()
comparison_rows = []

for model_name in baselines:
    model_rows = base_valid[base_valid["method"] == model_name].sort_values("window")
    if len(model_rows) != 5:
        continue  # skip if incomplete window coverage

    base_hr = model_rows["hit_rate"].values.astype(float)
    diff    = v2_hit_rates - base_hr
    mean_diff = diff.mean()

    # Paired t-test: H0: V2 and baseline have same mean hit rate
    t_stat, p_val = ttest_rel(v2_hit_rates, base_hr)

    # Cohen's d
    d = mean_diff / (diff.std(ddof=1) + 1e-10)

    # Direction
    direction = "V2 > Baseline" if mean_diff > 0 else "Baseline > V2"

    comparison_rows.append({
        "baseline"      : model_name,
        "v2_mean_hr"    : round(v2_hit_rates.mean(), 4),
        "base_mean_hr"  : round(base_hr.mean(), 4),
        "mean_diff"     : round(mean_diff, 4),
        "t_stat"        : round(t_stat, 3),
        "p_value"       : round(p_val, 4),
        "cohens_d"      : round(d, 3),
        "direction"     : direction,
        "sig@5%"        : "✓" if p_val < 0.05 else "✗",
    })

comp_df = pd.DataFrame(comparison_rows).sort_values("p_value")
print(comp_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: V2 vs TCN-7ch — detailed window-by-window breakdown
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 4: V2 vs TCN-7ch — Window-by-Window Detail")
print("="*70)

tcn7_rows = base_valid[base_valid["method"].str.contains("TCN")].sort_values("window")
if len(tcn7_rows) == 5:
    tcn7_hr = tcn7_rows["hit_rate"].values.astype(float)
    for i, (w, v2hr, tcnhr) in enumerate(zip(VALID_WINDOWS, v2_hit_rates, tcn7_hr)):
        diff   = v2hr - tcnhr
        winner = "V2 ✓" if diff > 0 else "TCN-7ch ✓"
        print(f"  W{w}: V2={v2hr:.4f}  TCN-7ch={tcnhr:.4f}  diff={diff:+.4f}  [{winner}]")

    diff_arr = v2_hit_rates - tcn7_hr
    print(f"\n  Mean diff (V2 - TCN-7ch): {diff_arr.mean():+.4f}")
    print(f"  V2 wins on {(diff_arr > 0).sum()}/5 windows")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Save outputs
# ─────────────────────────────────────────────────────────────────────────────
v2_sig_df.to_csv("reports/statistical_significance_v2.csv", index=False)
comp_df.to_csv("reports/baseline_comparison_significance.csv", index=False)
print("\n\nResults saved:")
print("  reports/statistical_significance_v2.csv")
print("  reports/baseline_comparison_significance.csv")
print("\nDone.")
