"""
Fair V1 vs V2 comparison -- apples-to-apples.

Runs the SAME walk-forward windows with:
  V1: 12ch Laplacian features + MarketTCN(12)
  V2: 16ch GAT + regime features + MarketTCN(16)

Primary metric: out-of-sample hit rate on confident trades (same as walk_forward.py).

Usage:
    python asset_evaluation/compare_v1_v2_fair.py           # full 5 windows (~30 min)
    python asset_evaluation/compare_v1_v2_fair.py --quick # 1 window smoke test
"""

import argparse
import os
import pickle
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from topo_trader.utils.data_loader import (
    fetch_universe_tickers,
    fetch_and_prepare_data,
    generate_features,
)
from topo_trader.train import train_model
from topo_trader.evaluation.walk_forward import run_walk_forward, WALK_FORWARD_WINDOWS


def load_features(version: str, tickers, force=False):
    cache = f"topo_trader/data/cache/features_{version}_meta.pkl"
    if not force and os.path.exists(cache):
        print(f"Loading cached {version.upper()} features...")
        with open(cache, "rb") as f:
            return pickle.load(f)

    data = fetch_and_prepare_data(tickers, start_date="2015-01-01", end_date="2024-01-01")
    features, _ = generate_features(data, tickers, parallel=False, n_jobs=1, version=version)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached {version.upper()} features -> {cache}")
    return features


def run_version(label, features, tickers, num_inputs, epochs, windows):
    def train_fn(X, y):
        return train_model(X, y, epochs=epochs, batch_size=64, lr=0.001, num_inputs=num_inputs)

    orig = WALK_FORWARD_WINDOWS.copy()
    try:
        if windows is not None:
            import topo_trader.evaluation.walk_forward as wf_mod
            wf_mod.WALK_FORWARD_WINDOWS = windows

        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        results_df, _ = run_walk_forward(
            ticker_features=features,
            tickers=tickers,
            train_model_fn=train_fn,
            window_len=64,
            verbose=True,
        )
        results_df["version"] = label
        return results_df
    finally:
        if windows is not None:
            import topo_trader.evaluation.walk_forward as wf_mod
            wf_mod.WALK_FORWARD_WINDOWS = orig


def print_side_by_side(v1_df, v2_df):
    cols = ["window", "regime_label", "test_period", "hit_rate", "overall_accuracy", "confident_trade_pct"]
    v1 = v1_df[cols].copy()
    v2 = v2_df[cols].copy()
    v1.columns = [f"v1_{c}" if c not in ("window", "regime_label", "test_period") else c for c in v1.columns]
    v2s = v2.rename(columns={
        "hit_rate": "v2_hit_rate",
        "overall_accuracy": "v2_overall_accuracy",
        "confident_trade_pct": "v2_confident_trade_pct",
    })
    merged = v1.merge(v2s[["window", "v2_hit_rate", "v2_overall_accuracy", "v2_confident_trade_pct"]], on="window")
    merged["hit_rate_delta"] = merged["v2_hit_rate"] - merged["v1_hit_rate"]
    merged["accuracy_delta"] = merged["v2_overall_accuracy"] - merged["v1_overall_accuracy"]

    print("\n=== SIDE-BY-SIDE WALK-FORWARD (OOS) ===")
    print(merged.to_string(index=False))

    print("\n=== SUMMARY ===")
    print(f"  V1 mean hit rate (confident): {v1_df['hit_rate'].mean():.1%}")
    print(f"  V2 mean hit rate (confident): {v2_df['hit_rate'].mean():.1%}")
    print(f"  Delta (V2 - V1):              {v2_df['hit_rate'].mean() - v1_df['hit_rate'].mean():+.1%}")
    print(f"  V2 wins on { (merged['hit_rate_delta'] > 0).sum() } / {len(merged)} windows")

    if v2_df["hit_rate"].mean() > v1_df["hit_rate"].mean():
        print("\n  >> V2 IMPROVED over V1 on average hit rate")
    elif v2_df["hit_rate"].mean() < v1_df["hit_rate"].mean():
        print("\n  >> V1 still ahead on average hit rate")
    else:
        print("\n  >> Tie on average hit rate")


def main():
    parser = argparse.ArgumentParser(description="Fair V1 vs V2 walk-forward comparison")
    parser.add_argument("--quick", action="store_true", help="Run only window 3 (COVID) for smoke test")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per window (default 10)")
    parser.add_argument("--force-features", action="store_true", help="Regenerate feature cache")
    args = parser.parse_args()

    tickers = fetch_universe_tickers()
    windows = [WALK_FORWARD_WINDOWS[2]] if args.quick else None  # COVID window only in quick mode

    feat_v1 = load_features("v1", tickers, force=args.force_features)
    feat_v2 = load_features("v2", tickers, force=args.force_features)

    v1_results = run_version("V1 (12ch Laplacian)", feat_v1, tickers, num_inputs=12, epochs=args.epochs, windows=windows)
    v2_results = run_version("V2 (16ch GAT+Regime)", feat_v2, tickers, num_inputs=16, epochs=args.epochs, windows=windows)

    print_side_by_side(v1_results, v2_results)

    os.makedirs("reports", exist_ok=True)
    v1_results.to_csv("reports/walk_forward_v1.csv", index=False)
    v2_results.to_csv("reports/walk_forward_v2_rerun.csv", index=False)
    pd.concat([v1_results, v2_results]).to_csv("reports/walk_forward_v1_v2_combined.csv", index=False)
    print("\nSaved: reports/walk_forward_v1.csv, reports/walk_forward_v2_rerun.csv")


if __name__ == "__main__":
    main()
