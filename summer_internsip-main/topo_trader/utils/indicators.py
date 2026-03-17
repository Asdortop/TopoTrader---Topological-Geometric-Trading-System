import numpy as np
import pandas as pd

def get_log_returns(prices):
    """
    Channel 1: Logarithmic Returns
    r_t = ln(P_t / P_{t-1})
    """
    return np.log(prices / prices.shift(1)).fillna(0)

def get_normalized_volume(volume):
    """
    Channel 2: Normalized Volume
    v_t = ln((V_t + 1) / (V_{t-1} + 1))
    """
    return np.log((volume + 1) / (volume.shift(1) + 1)).fillna(0)

def get_rsi(prices, period=14):
    """
    Channel 3: RSI
    Scaled to [0, 1] by dividing by 100.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return (rsi / 100.0).fillna(0)

def get_macd_normalized(prices, fast=12, slow=26, signal=9, norm_window=64):
    """
    Channel 4: MACD
    Normalized via Z-score over rolling window.
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    # Z-score normalization
    rolling_mean = macd_hist.rolling(window=norm_window).mean()
    rolling_std = macd_hist.rolling(window=norm_window).std()
    
    # Avoid division by zero
    z_score = (macd_hist - rolling_mean) / (rolling_std + 1e-8)
    return z_score.fillna(0)

def get_atr_normalized(high, low, close, period=14):
    """
    Channel 5: Average True Range (ATR)
    Normalized by Close Price.
    """
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    natr = atr / close
    return natr.fillna(0)

def get_bollinger_b(prices, period=20, std_dev=2):
    """
    Channel 6: Bollinger Band %B
    """
    sma = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper = sma + (rolling_std * std_dev)
    lower = sma - (rolling_std * std_dev)
    
    pct_b = (prices - lower) / (upper - lower + 1e-8)
    return pct_b.fillna(0)

def get_price_zscore(prices, period=50):
    """
    Channel 7: Z-Score of Price (Detrending)
    """
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    z_score = (prices - sma) / (std + 1e-8)
    return z_score.fillna(0)
