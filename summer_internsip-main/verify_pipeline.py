import os
import torch
import numpy as np
from topo_trader.utils.data_loader import fetch_and_prepare_data, generate_features, create_dataset
from topo_trader.train import train_model
from topo_trader.backtester import backtest_logic

def run_verification():
    print("=== Starting Verification Pipeline ===")
    
    # 1. Use a very small subset of tickers for speed
    tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]
    print(f"1. Fetching data for {tickers}...")
    
    try:
        # Short timeframe
        data = fetch_and_prepare_data(tickers, start_date="2023-01-01", end_date="2023-06-01")
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Data fetch failed: {e}")
        return

    # 2. Generate Features
    try:
        features, _ = generate_features(data, tickers)
        print("Feature generation successful.")
        
        # Check one ticker
        df = features["AAPL"]
        print(f"AAPL Feature Columns: {df.columns.tolist()}")
        print(f"AAPL Shape: {df.shape}")
        
    except Exception as e:
        print(f"Feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Create Tensor Dataset
    try:
        X, y = create_dataset(features, tickers, window_len=64)
        print(f"Dataset created. X shape: {X.shape}, y shape: {y.shape}")
        
        if len(X) == 0:
            print("Dataset is empty! Check window length vs data duration.")
            return
            
    except Exception as e:
        print(f"Tensor creation failed: {e}")
        traceback.print_exc()
        return

    # 4. Train Model (Dummy run)
    try:
        model = train_model(X, y, epochs=1, batch_size=4)
        print("Model training successful.")
    except Exception as e:
        print(f"Model training failed: {e}")
        traceback.print_exc()
        return

    # 5. Backtest Logic Check
    try:
        sample_tensor = X[0] # (12, 64)
        # Fake inputs
        vol = 0.02
        h0 = 2.5
        
        print("Running backtest check...")
        pos = backtest_logic(model, sample_tensor, vol, capital=10000, h0_entropy=h0)
        print(f"Backtest Position Size: {pos}")
        
    except Exception as e:
        print(f"Backtest logic failed: {e}")
        traceback.print_exc()
        return

    print("=== Verification Complete: SUCCESS ===")

if __name__ == "__main__":
    run_verification()
