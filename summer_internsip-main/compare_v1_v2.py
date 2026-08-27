"""
Head-to-head V1 vs V2 comparison (fair methodology).

Rules for a valid comparison:
  1. Same data, same tickers, same date windows
  2. V1: 12ch Laplacian features  + fresh TCN per window (num_inputs=12)
  3. V2: 16ch GAT + regime features + fresh TCN per window (num_inputs=16)
  4. Walk-forward: train on past, test on future (true out-of-sample)
  5. Same metrics: hit rate on confident trades (prob > 0.55 or < 0.45)

Usage:
    python compare_v1_v2.py              # walk-forward only (~20-30 min)
    python compare_v1_v2.py --quick      # 3 epochs, 3 windows (~5 min smoke test)
"""

import argparse
import os
import pickle
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from topo_trader.utils.data_loader import (
    fetch_universe_tickers,
    fetch_and_prepare_data,
    generate_features,
)
from topo_trader.train import train_model
from topo_trader.evaluation.walk_forward import (
    WALK_FORWARD_WINDOWS,
    create_dataset_for_range,
    evaluate_model,
)


def load_or_build_features(version: str, tickers):
    cache = f"topo_trader/data/cache/features_{version}_meta.pkl"
    if os.path.exists(cache):
        print(f"Loading cached {version.upper()} features from {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)
    data = fetch_and_prepare_data(tickers, "2015-01-01", "2024-01-01")
    features, _ = generate_features(data, tickers, parallel=False, n_jobs=1, version=version)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached {version.upper()} features to {cache}")
    return features


def run_version_walk_forward(features, tickers, version: str, epochs: int, windows):
    num_inputs = 12 if version == "v1" else 16

    def train_fn(X, y):
        return train_model(X, y, epochs=epochs, batch_size=64, lr=0.001, num_inputs=num_inputs)

    rows = []
    for i, (train_s, train_e, test_s, test_e, label) in enumerate(windows):
        print(f"\n[{version.upper()}] Window {i+1}/{len(windows)} -- {label}")
        X_train, y_train = create_dataset_for_range(features, tickers, 64, train_s, train_e)
        X_test, y_test = create_dataset_for_range(features, tickers, 64, test_s, test_e)
        if len(X_train) == 0 or len(X_test) == 0:
            print("  SKIP -- insufficient data")
            continue
        print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")
        model = train_fn(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({
            "version": version,
            "window": i + 1,
            "regime_label": label,
            "test_period": f"{test_s} -> {test_e}",
            **metrics,
        })
        print(f"  Hit rate: {metrics['hit_rate']:.3f}  "
              f"(confident {metrics['confident_trade_pct']:.1%})")
    return pd.DataFrame(rows)


def print_summary(v1_df: pd.DataFrame, v2_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE WALK-FORWARD COMPARISON (out-of-sample)")
    print("=" * 70)

    merged = v1_df.merge(
        v2_df,
        on=["window", "regime_label", "test_period"],
        suffixes=("_v1", "_v2"),
    )
    display = merged[[
        "window", "regime_label",
        "hit_rate_v1", "hit_rate_v2",
        "confident_trade_pct_v1", "confident_trade_pct_v2",
    ]].copy()
    display["delta_hit_rate"] = display["hit_rate_v2"] - display["hit_rate_v1"]
    display.columns = [
        "Win", "Regime", "V1 Hit%", "V2 Hit%", "V1 Conf%", "V2 Conf%", "V2-V1"
    ]
    print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    v1_mean = v1_df["hit_rate"].mean()
    v2_mean = v2_df["hit_rate"].mean()
    print(f"\nMean hit rate (confident trades):")
    print(f"  V1: {v1_mean:.3f}")
    print(f"  V2: {v2_mean:.3f}")
    print(f"  V2 - V1: {v2_mean - v1_mean:+.3f}  ({'V2 improved' if v2_mean > v1_mean else 'V1 better'})")

    v1_wins = (display["V2-V1"] < 0).sum()  # wait, delta = v2 - v1, v1 wins when delta < 0
    v2_wins = (display["V2-V1"] > 0).sum()
    print(f"\nWindows where V2 beats V1: {v2_wins}/{len(display)}")


def main():
    parser = argparse.ArgumentParser(description="Fair V1 vs V2 walk-forward comparison")
    parser.add_argument("--quick", action="store_true", help="3 epochs, first 3 windows only")
    args = parser.parse_args()

    epochs = 3 if args.quick else 10
    windows = WALK_FORWARD_WINDOWS[:3] if args.quick else WALK_FORWARD_WINDOWS

    print("=== Fair V1 vs V2 Comparison ===")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}  |  Epochs/window: {epochs}")
    print("V1 = 12ch Laplacian  |  V2 = 16ch GAT + regime\n")

    tickers = fetch_universe_tickers()
    features_v1 = load_or_build_features("v1", tickers)
    features_v2 = load_or_build_features("v2", tickers)

    v1_results = run_version_walk_forward(features_v1, tickers, "v1", epochs, windows)
    v2_results = run_version_walk_forward(features_v2, tickers, "v2", epochs, windows)

    os.makedirs("reports", exist_ok=True)
    v1_results.to_csv("reports/walk_forward_v1_compare.csv", index=False)
    v2_results.to_csv("reports/walk_forward_v2_compare.csv", index=False)

    combined = pd.concat([v1_results, v2_results], ignore_index=True)
    combined.to_csv("reports/compare_v1_v2_walkforward.csv", index=False)

    print_summary(v1_results, v2_results)
    print("\nSaved: reports/compare_v1_v2_walkforward.csv")


if __name__ == "__main__":
    main()
