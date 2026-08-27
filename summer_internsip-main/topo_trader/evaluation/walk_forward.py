"""
TopoTrader V2 - Walk-Forward Validation Framework
==================================================
6 rolling windows covering bull, bear, COVID crash, rate hike cycle.
"""

import traceback
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

# Walk-Forward Window Definitions — US Market (S&P 500 universe, 2015–2023)
WALK_FORWARD_WINDOWS = [
    ("2015-01-01", "2017-12-31", "2018-01-01", "2018-12-31", "Stable Bull"),
    ("2015-01-01", "2018-12-31", "2019-01-01", "2019-12-31", "Q4-2018 Recovery"),
    ("2015-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "COVID Crash"),
    ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "Post-COVID Bull"),
    ("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "Rate Hike Selloff"),
    ("2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "Soft Landing"),
]

# Walk-Forward Window Definitions — India Market (Nifty-50 universe, 2010–2021)
# Uses full available Kaggle data range; all 49 tickers have data from 2010.
# Key regimes: GFC recovery, Demonetization (2016), GST (2017),
#              IL&FS crisis (2018), COVID crash (2020).
INDIA_WALK_FORWARD_WINDOWS = [
    ("2010-01-01", "2013-12-31", "2014-01-01", "2014-12-31", "Post-GFC Recovery"),
    ("2010-01-01", "2014-12-31", "2015-01-01", "2015-12-31", "China Slowdown"),
    ("2010-01-01", "2015-12-31", "2016-01-01", "2016-12-31", "Demonetization"),
    ("2010-01-01", "2016-12-31", "2017-01-01", "2017-12-31", "GST Launch"),
    ("2010-01-01", "2017-12-31", "2018-01-01", "2018-12-31", "IL&FS Crisis"),
    ("2010-01-01", "2018-12-31", "2019-01-01", "2019-12-31", "Elections/Slowdown"),
    ("2010-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "COVID Crash"),
    ("2010-01-01", "2020-12-31", "2021-01-01", "2021-04-30", "Post-COVID Bull"),
]


def create_dataset_for_range(ticker_features, tickers, window_len,
                              start_date, end_date, channel_stats=None):
    """
    Create (N, n_channels, window_len) tensors for a specific date range.
    Compatible with both 12-channel (V1) and 16-channel (V2) feature sets.

    Per-channel Z-score normalization fixes the scale mismatch between
    heterogeneous channels (RSI [0,100] vs GAT [-1,1] vs H0 [0,5]).

    Args:
        channel_stats: (mean, std) arrays from training set. If None, computes
                       from this range (training call). If provided, uses those
                       stats (test call — no data leakage).
    Returns:
        X     : (N, C, T) float32, normalized
        y     : (N,) float32
        stats : (ch_mean, ch_std) — pass to test call
    """
    import sys
    all_X, all_y = [], []
    n_skipped = 0

    for ticker in tickers:
        if ticker not in ticker_features:
            continue

        try:
            df       = ticker_features[ticker]
            # Use boolean mask to avoid any .loc DatetimeIndex edge cases
            mask     = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
            df_range = df.loc[mask]

            if len(df_range) < window_len + 1:
                n_skipped += 1
                continue

            if "C1_LogRet" not in df_range.columns:
                n_skipped += 1
                continue

            # Force numeric dtype; skip ticker if any column can't be cast
            data_values = pd.to_numeric(
                df_range.values.ravel(), errors='coerce'
            ).reshape(df_range.shape).astype(np.float32)

            # Replace any remaining NaN/inf with 0
            data_values = np.nan_to_num(data_values, nan=0.0, posinf=0.0, neginf=0.0)

            targets   = (df_range["C1_LogRet"].shift(-1) > 0).astype(int).values
            n_samples = len(data_values) - window_len - 1

            if n_samples <= 0:
                n_skipped += 1
                continue

            for i in range(n_samples):
                x_window = data_values[i : i + window_len].T.copy()  # force copy, no view
                y_label  = float(targets[i + window_len - 1])
                all_X.append(x_window)
                all_y.append(y_label)

        except Exception as e:
            print(f"  [WARN] Skipping ticker {ticker}: {e}", flush=True)
            traceback.print_exc()
            n_skipped += 1
            continue

    if not all_X:
        n_ch = next(iter(ticker_features.values())).shape[1] if ticker_features else 16
        print(f"  [create_dataset] 0 samples built, {n_skipped} tickers skipped", flush=True)
        # Stats shape must match N_PRICE_CH=7 (what the normalization block reads back)
        empty_stats = (np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32))
        return (np.empty((0, n_ch, window_len), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                empty_stats)

    X = np.stack(all_X, axis=0)   # (N, C, T)
    y = np.array(all_y, dtype=np.float32)

    # ── Selective Z-score normalization: C1–C7 only ────────────────────────
    N_PRICE_CH = 7  # normalize only price/technical channels

    if channel_stats is None:
        ch_mean = X[:, :N_PRICE_CH, :].mean(axis=(0, 2))   # (7,) — train call
        ch_std  = X[:, :N_PRICE_CH, :].std(axis=(0, 2))
    else:
        ch_mean, ch_std = channel_stats                     # (7,) — test call (no leakage)

    safe_std = np.where(ch_std < 1e-8, 1.0, ch_std)
    X_norm = X.copy()
    X_norm[:, :N_PRICE_CH, :] = (
        (X[:, :N_PRICE_CH, :] - ch_mean[np.newaxis, :, np.newaxis])
        / safe_std[np.newaxis, :, np.newaxis]
    )
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return X_norm, y, (ch_mean, ch_std)

def evaluate_model(model, X_test, y_test, long_thresh=0.55, short_thresh=0.45):
    """
    Evaluate a trained model on the test set.
    Automatically moves tensors to the same device as the model (CPU or CUDA).
    """
    # Detect model device from its parameters
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X_test, dtype=torch.float32).to(device)
        probs = model(X_t).flatten().cpu().numpy()

    preds           = (probs > 0.5).astype(int)
    overall_acc     = accuracy_score(y_test.astype(int), preds)

    confident_mask  = (probs > long_thresh) | (probs < short_thresh)
    if confident_mask.sum() > 0:
        hit_rate    = accuracy_score(y_test[confident_mask].astype(int),
                                     preds[confident_mask])
        conf_pct    = confident_mask.mean()
    else:
        hit_rate    = 0.0
        conf_pct    = 0.0

    return {
        "overall_accuracy"     : round(overall_acc, 4),
        "hit_rate"             : round(hit_rate, 4),
        "confident_trade_pct"  : round(conf_pct, 4),
        "n_test"               : len(X_test),
    }


def run_walk_forward(ticker_features, tickers, train_model_fn,
                     window_len=64, verbose=True, windows=None):
    """
    Run walk-forward validation.

    Args:
        ticker_features: dict of {ticker: DataFrame} from generate_features().
        tickers: list of ticker strings.
        train_model_fn: callable(X_train, y_train) -> trained model.
        window_len: sequence length for TCN input (default 64).
        verbose: print progress.
        windows: list of (train_s, train_e, test_s, test_e, label) tuples.
                 Defaults to WALK_FORWARD_WINDOWS (US) if None.

    Returns:
        results_df: pd.DataFrame with metrics per window.
        models: list of trained models (one per window).
    """
    if windows is None:
        windows = WALK_FORWARD_WINDOWS

    results = []
    models  = []
    n_total = len(windows)

    for i, (train_s, train_e, test_s, test_e, label) in enumerate(windows):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Walk-Forward Window {i+1}/{n_total} -- {label}")
            print(f"  Train : {train_s}  ->  {train_e}")
            print(f"  Test  : {test_s}  ->  {test_e}")

        try:
            X_train, y_train, train_stats = create_dataset_for_range(
                ticker_features, tickers, window_len, train_s, train_e
            )
            # Pass train stats to test — same scale, no data leakage
            X_test,  y_test,  _           = create_dataset_for_range(
                ticker_features, tickers, window_len, test_s, test_e,
                channel_stats=train_stats
            )
        except Exception:
            print("  ERROR in create_dataset_for_range:")
            traceback.print_exc()
            continue

        if len(X_train) == 0:
            if verbose: print("  SKIP -- insufficient training data")
            continue
        if len(X_test) == 0:
            if verbose: print("  SKIP -- insufficient test data")
            continue

        if verbose:
            print(f"  Train samples: {len(X_train):,}   Test samples: {len(X_test):,}")

        try:
            model  = train_model_fn(X_train, y_train)
        except Exception:
            print("  ERROR in train_model_fn:")
            traceback.print_exc()
            continue
        models.append(model)

        try:
            metrics = evaluate_model(model, X_test, y_test)
        except Exception:
            print("  ERROR in evaluate_model:")
            traceback.print_exc()
            continue

        row = {
            "window"             : i + 1,
            "regime_label"       : label,
            "test_period"        : f"{test_s} -> {test_e}",
            "n_train"            : len(X_train),
            **metrics,
        }
        results.append(row)

        if verbose:
            print(f"  Overall Acc    : {metrics['overall_accuracy']:.3f}")
            print(f"  Hit Rate       : {metrics['hit_rate']:.3f}   (confident trades only)")
            print(f"  Confident Pct  : {metrics['confident_trade_pct']:.1%} of predictions")

    results_df = pd.DataFrame(results)

    if verbose and len(results_df) > 0:
        print(f"\n{'='*60}")
        print("Walk-Forward Summary")
        print(results_df[["window", "regime_label", "test_period",
                           "hit_rate", "confident_trade_pct"]].to_string(index=False))
        print(f"\nMean Hit Rate : {results_df['hit_rate'].mean():.3f}")
        print(f"Std Hit Rate  : {results_df['hit_rate'].std():.3f}")

    return results_df, models
