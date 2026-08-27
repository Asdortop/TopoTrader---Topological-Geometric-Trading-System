"""
TopoTrader V2 -- Full Pipeline Runner
=====================================
Upgrades over V1 (run_training_full.py):
  - V2 feature set: 16 channels (C8 = GAT signal, C13-C16 = regime one-hot)
  - Walk-forward validation across 6 windows (6 years of out-of-sample tests)
  - Crash Forecaster training (separate LSTM on TDA entropy trajectory)
  - Indian market (Nifty 200) support via fetch_india_tickers()

Usage:
    # US market (default)
    python run_training_v2.py

    # Indian market
    python run_training_v2.py --market india
"""

import argparse
import torch
import numpy as np
import os

from topo_trader.utils.data_loader import (
    fetch_universe_tickers,
    fetch_india_csv_tickers,
    load_india_csv_data,
    fetch_and_prepare_data,
    generate_features,
)
from topo_trader.train import train_model, save_model
from topo_trader.evaluation.walk_forward import (
    run_walk_forward, create_dataset_for_range,
    WALK_FORWARD_WINDOWS, INDIA_WALK_FORWARD_WINDOWS,
)
from topo_trader.models.crash_forecaster import train_crash_forecaster, predict_crash_probability


# -- Quick dataset helper for walk-forward -------------------------------------
def _make_dataset(X_train, y_train, epochs=10, batch_size=64, lr=0.001):
    """Adapter: walk_forward passes (X, y) and expects a model back."""
    return train_model(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)


def run_pipeline(market: str = "us"):
    print(f"=== TopoTrader V2 -- {market.upper()} Market Pipeline ===")

    # -- 1. Universe ------------------------------------------------------------
    if market == "india":
        tickers    = fetch_india_csv_tickers()   # from local Kaggle CSVs
        cache_name = "india_csv_data.parquet"
        spy_proxy  = ""                           # no index CSV, use mean returns fallback
        model_name = "topo_trader/models/tcn_v2_india.pth"
        crash_name = "topo_trader/models/crash_forecaster_india.pth"
    else:
        tickers    = fetch_universe_tickers()
        cache_name = "raw_market_data.parquet"
        spy_proxy  = "SPY"
        model_name = "topo_trader/models/tcn_v2_us.pth"
        crash_name = "topo_trader/models/crash_forecaster_us.pth"

    print(f"Universe: {len(tickers)} assets  |  Market proxy: {spy_proxy}")

    # -- 2. Data ----------------------------------------------------------------
    if market == "india":
        data    = load_india_csv_data(
            start_date="2010-01-01",
            end_date="2021-04-30",
            cache_name=cache_name,
        )
        tickers = list(data.columns.get_level_values(0).unique())  # use tickers found in CSVs
    else:
        data = fetch_and_prepare_data(
            tickers,
            start_date="2015-01-01",
            end_date="2024-01-01",
            force_reload=False,
            cache_name=cache_name,
        )

    # -- 3. Feature Generation (V2: 16 channels) --------------------------------
    features, common_index = generate_features(data, tickers, parallel=False, n_jobs=1)

    if not features:
        print("ERROR: No features generated -- no matching tickers found in downloaded data.")
        print("Tip: Delete the cache file and retry to force a fresh download.")
        return None, None, None

    print(f"\nFeature shape per ticker: {next(iter(features.values())).shape}")
    print(f"Channels: {list(next(iter(features.values())).columns)}")

    # -- 4. Walk-Forward Validation -----------------------------------------------
    windows = INDIA_WALK_FORWARD_WINDOWS if market == "india" else WALK_FORWARD_WINDOWS
    n_windows = len(windows)
    print(f"\n=== Walk-Forward Validation ({n_windows} windows) ===")
    results_df, wf_models = run_walk_forward(
        ticker_features=features,
        tickers=tickers,
        train_model_fn=lambda X, y: _make_dataset(X, y, epochs=30),
        window_len=64,
        verbose=True,
        windows=windows,
    )

    # Save walk-forward results
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv(f"reports/walk_forward_{market}.csv", index=False)
    print(f"\nWalk-forward results saved to reports/walk_forward_{market}.csv")

    # -- 5. Final Model: Train on full available data, Validate on last year ---
    print("\n=== Training Final Model (full train -> last year validation) ===")
    X_train, y_train, _train_stats = create_dataset_for_range(
        features, tickers, 64, "2015-01-01", "2022-12-31"
    )
    X_val, y_val, _ = create_dataset_for_range(
        features, tickers, 64, "2023-01-01", "2023-12-31",
        channel_stats=_train_stats
    )
    print(f"Train: {len(X_train):,} samples  |  Val: {len(X_val):,} samples")

    final_model = None
    if len(X_train) > 0:
        final_model = train_model(X_train, y_train, epochs=20, batch_size=64, lr=0.001)

        if len(X_val) > 0:
            device = next(final_model.parameters()).device
            final_model.eval()
            with torch.no_grad():
                probs = final_model(
                    torch.tensor(X_val, dtype=torch.float32).to(device)
                ).flatten().cpu().numpy()
            preds   = (probs > 0.5).astype(int)
            val_acc = (preds == y_val).mean()
            conf    = ((probs > 0.55) | (probs < 0.45)).mean()
            conf_acc = (preds[(probs > 0.55) | (probs < 0.45)] ==
                        y_val[(probs > 0.55) | (probs < 0.45)]).mean() if conf > 0 else 0
            print(f"Val accuracy: {val_acc:.3f}")
            print(f"Hit rate (confident only, {conf:.1%} of trades): {conf_acc:.3f}")
        else:
            print("No validation data available (expected for India dataset ending 2021)")

        save_model(final_model, model_name)
    else:
        print("ERROR: No training data generated. Check date ranges and data.")


    # -- 6. Crash Forecaster Training -------------------------------------------
    print("\n=== Training Crash Probability Forecaster ===")
    # Extract market-wide H0/H1 from any ticker (they are all the same)
    ref_ticker  = next(iter(features))
    ref_df      = features[ref_ticker]
    h0_vec      = ref_df["C10_H0"].values
    h1_vec      = ref_df["C11_H1"].values

    # Market return proxy
    if spy_proxy in features:
        market_ret = features[spy_proxy]["C1_LogRet"].values
    else:
        # Fallback: mean of all returns
        market_ret = np.mean(
            [features[t]["C1_LogRet"].values for t in features], axis=0
        )

    # Train end index: last day of 2022
    train_end_idx = int((ref_df.index <= "2022-12-31").sum()) - 1

    crash_model = train_crash_forecaster(
        h0_vec=h0_vec,
        h1_vec=h1_vec,
        market_returns=market_ret,
        train_end_idx=train_end_idx,
        window=30,
        epochs=40,
        lr=5e-4,
        save_path=crash_name,
    )

    # Sample crash probability for most recent 30 days
    recent_probs = predict_crash_probability(crash_model, h0_vec, h1_vec, market_ret)
    print(f"\nCurrent crash probabilities (most recent data):")
    print(f"  P(crash in  5 days): {recent_probs[0]:.2%}")
    print(f"  P(crash in 10 days): {recent_probs[1]:.2%}")
    print(f"  P(crash in 20 days): {recent_probs[2]:.2%}")

    print("\n=== Pipeline Complete ===")
    return final_model, crash_model, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TopoTrader V2 Pipeline")
    parser.add_argument(
        "--market",
        type=str,
        default="us",
        choices=["us", "india"],
        help="Market to run (default: us)",
    )
    args = parser.parse_args()

    # On Windows, protect multiprocessing entry point
    from joblib import parallel_backend
    run_pipeline(market=args.market)
