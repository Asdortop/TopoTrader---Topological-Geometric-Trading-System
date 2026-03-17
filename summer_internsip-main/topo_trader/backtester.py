import numpy as np
import torch

def backtest_logic(model, data_tensor, current_volatility, capital=100000, h0_entropy=None):
    """
    7. Phase 5: Risk Management and Execution Logic
    
    Args:
        model: Trained PyTorch TCN model.
        data_tensor: (1, 12, 64) Input tensor for the current day.
        current_volatility: Current Normalized ATR (Scalar) for the asset.
        capital: Allocated capital per asset (or total capital, dependent on portfolio logic).
                 Here we assume capital is the max allocation for THIS asset.
        h0_entropy: Current H0 persistence entropy for the regime veto.
        
    Returns:
        signed_position_size: Dollar amount to go Long (+) or Short (-).
    """
    model.eval()
    
    # 1. Topological Veto (Regime Filter)
    # Rule: If H0_Entropy < Crash_Threshold (e.g. 1.5 sigma below mean), Halt.
    # We need historical mean/std of entropy to do this properly.
    # For this implementation, we will use a fixed hardcoded threshold or skip if None.
    # Let's assume a threshold passed or derived. 
    # Example heuristic: if entropy < 1.0 (assuming it ranges ~2-5 usually).
    
    CRASH_THRESHOLD = 1.0 # Placeholder
    if h0_entropy is not None and h0_entropy < CRASH_THRESHOLD:
        return 0.0 # Cash
        
    # 2. Prediction
    with torch.no_grad():
        # Input shape needs to be (1, 12, 64)
        if not isinstance(data_tensor, torch.Tensor):
            data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
            
        if len(data_tensor.shape) == 2:
            data_tensor = data_tensor.unsqueeze(0)
            
        prob = model(data_tensor).item()
        
    # 3. Deadband Filter
    # Long: > 0.55, Short: < 0.45, else Neutral
    signal = 0
    
    if prob > 0.55:
        signal = 1
    elif prob < 0.45:
        signal = -1
    else:
        signal = 0
        
    if signal == 0:
        return 0.0
        
    # 4. Volatility Scaling (Inverse Volatility Sizing)
    # Size = (Target Vol / Current Vol) * Capital
    TARGET_VOL = 0.20 # 20%
    
    # Avoid div/0
    if current_volatility <= 0:
        vol_factor = 0
    else:
        vol_factor = TARGET_VOL / current_volatility
        
    # Cap leverage if needed (e.g. max 2x)
    vol_factor = min(vol_factor, 2.0)
    
    position_size = vol_factor * capital
    
    return signal * position_size
