import torch
import numpy as np
from topo_trader.utils.data_loader import (
    fetch_universe_tickers,
    fetch_and_prepare_data,
    generate_features,
)
from topo_trader.train import train_model, save_model


def create_dataset_for_range(
    ticker_features, tickers, window_len=64, start_date="2015-01-01", end_date="2023-12-31"
):
    """
    Create (N, 12, window_len) tensors for a specific date range.
    """
    all_X = []
    all_y = []

    for ticker in tickers:
        if ticker not in ticker_features:
            continue

        df = ticker_features[ticker]
        df_range = df.loc[start_date:end_date]

        data_values = df_range.values
        if "C1_LogRet" not in df_range.columns:
            continue

        targets = (df_range["C1_LogRet"].shift(-1) > 0).astype(int).values

        n_samples = len(data_values) - window_len - 1
        if n_samples <= 0:
            continue

        for i in range(n_samples):
            x_window = data_values[i : i + window_len].T
            y_label = float(targets[i + window_len - 1])
            all_X.append(x_window)
            all_y.append(y_label)

    if not all_X:
        return np.empty((0, 12, window_len)), np.empty((0,))

    return np.array(all_X), np.array(all_y)


def run_pipeline():
    print("=== Starting Full Scale Geometric Trading Pipeline ===")

    # 1. Define Expanded Universe
    tickers = fetch_universe_tickers()
    print(f"Target Universe: {len(tickers)} assets.")

    # 2. Fetch Data (Cached)
    # Use history from 2015 up to the end of 2023 for features.
    # end_date is exclusive -> up to 2023-12-31.
    data = fetch_and_prepare_data(
        tickers, start_date="2015-01-01", end_date="2024-01-01"
    )

    # 3. Generate Features (Parallelized)
    features, _ = generate_features(data, tickers, parallel=True, n_jobs=-1)

    # 4. Create Train / Validation Datasets (both inside 2015-2023)
    X_train, y_train = create_dataset_for_range(
        features,
        tickers,
        window_len=64,
        start_date="2015-01-01",
        end_date="2021-12-31",
    )
    X_val, y_val = create_dataset_for_range(
        features,
        tickers,
        window_len=64,
        start_date="2022-01-01",
        end_date="2023-12-31",
    )

    print(f"Train Dataset Shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation Dataset Shape: X={X_val.shape}, y={y_val.shape}")

    # 5. Train Model on Train Set
    if len(X_train) > 0:
        model = train_model(X_train, y_train, epochs=5, batch_size=64, lr=0.001)

        # Simple validation accuracy (directional hit rate) for logging
        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                probs = model(X_val_tensor).flatten().numpy()
            preds = (probs > 0.5).astype(int)
            val_acc = (preds == y_val).mean()
            print(f"Validation directional accuracy (2022-2023): {val_acc:.3f}")

        # 6. Save
        save_model(model, "topo_trader/models/tcn_full.pth")
    else:
        print("Error: Train dataset is empty.")


if __name__ == "__main__":
    # Ensure parallel processing works safely on Windows
    from joblib import parallel_backend

    # On Windows, joblib sometimes needs protection for multiprocessing
    # But usually threading backend is safer for simple numpy/scipy tasks if GIL is released,
    # however ripser is C++ bound so it might release GIL.
    # Default 'loky' is robust.
    run_pipeline()
