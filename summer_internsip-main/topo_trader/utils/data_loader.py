import yfinance as yf
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from .indicators import (
    get_log_returns, get_normalized_volume, get_rsi, get_macd_normalized,
    get_atr_normalized, get_bollinger_b, get_price_zscore
)
from ..strategies.graph_engine import get_laplacian_residuals
from ..strategies.topology_engine import get_tda_features
from ..strategies.walsh_filter import get_walsh_score

CACHE_DIR = "topo_trader/data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_universe_tickers():
    # Expanded Universe: S&P 100 + Major Tech + Liquid ETFs
    # Ideally we'd scrape Wikipedia for S&P 500, but for stability we use a larger static list.
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
        "JPM", "XOM", "V", "PG", "HD", "MA", "CVX", "ABBV", "PEP", "KO",
        "LLY", "BAC", "COST", "MRK", "AVGO", "TMO", "DIS", "PFE", "CSCO", "ACN", "SPY",
        "QQQ", "IWM", "DIA", "EEM", "TLT", "GLD", "SLV", "USO", "UNG",
        "AMD", "INTC", "QCOM", "TXN", "HON", "UNP", "LIN", "PM", "IBM", "AMGN",
        "CAT", "GS", "BA", "MMM", "GE", "RTX", "LMT", "DE", "CAT"
    ]
    # Deduplicate
    tickers = list(set(tickers))
    return tickers

def fetch_and_prepare_data(tickers, start_date="2015-01-01", end_date="2024-01-01", force_reload=False):
    cache_path = os.path.join(CACHE_DIR, "raw_market_data.parquet")
    
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Downloading data for {len(tickers)} tickers...")
    
    # Download in bulk
    # yfinance 'auto_adjust=True' gets OHLC adjusted for splits/divs
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    
    # Basic validation and forward fill
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    
    # Save to cache
    try:
        data.to_parquet(cache_path)
        print(f"Data cached to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not cache data: {e}")
    
    return data

def process_time_step(t, lookback, asset_list, returns_matrix, walsh_lookback, laplacian_matrix_ref=None):
    """
    Helper function for parallel processing a single time step.
    Warning: Parallelizing properly requires sharing the state (laplacian history) which makes it tricky.
    
    The Geometric features depend on the window [t-lookback : t]. This is independent per step t.
    HOWEVER, Walsh score depends on history of Residuals. 
    So we first need to compute ALL Residuals (can be parallelized per t), 
    THEN compute Walsh scores (requires sequence of residuals).
    """
    # Use returns_matrix which is read-only shared memory usually in joblib
    # Window (Assets, Lookback)
    window = returns_matrix[t-lookback:t, :].T
    
    # 1. Laplacian Residuals
    residuals = get_laplacian_residuals(window, threshold=0.5)
    
    # 2. Topology
    topo = get_tda_features(window)
    
    return {
        't': t,
        'residuals': residuals,
        'h0': topo[0],
        'h1': topo[1]
    }

def generate_features(data, tickers, parallel=True, n_jobs=-1):
    print("Generating features...")
    
    # Structure: features[ticker] = DataFrame
    ticker_features = {}
    
    # Global Market Proxy (SPY)
    spy_ticker = "SPY"
    if spy_ticker in tickers and spy_ticker in data.columns.levels[0]:
        spy_close = data[spy_ticker]['Close']
        spy_log_ret = get_log_returns(spy_close)
    else:
        spy_log_ret = pd.Series(0, index=data.index)

    # 1. Compute Standard Indicators (Fast, Vectorized)
    print("Computing Standard Indicators...")
    for ticker in tqdm(tickers):
        if ticker not in data.columns.levels[0]:
            continue
            
        df = data[ticker].copy()
        
        # Base Features
        c1 = get_log_returns(df['Close'])
        c2 = get_normalized_volume(df['Volume'])
        c3 = get_rsi(df['Close'])
        c4 = get_macd_normalized(df['Close'])
        c5 = get_atr_normalized(df['High'], df['Low'], df['Close'])
        c6 = get_bollinger_b(df['Close'])
        c7 = get_price_zscore(df['Close'])
        c12 = spy_log_ret
        
        # Prepare DataFrame
        feat_df = pd.DataFrame({
            'C1_LogRet': c1,
            'C2_Vol': c2,
            'C3_RSI': c3,
            'C4_MACD': c4,
            'C5_ATR': c5,
            'C6_BB': c6,
            'C7_ZScore': c7,
            'C8_Laplacian': 0.0,
            'C9_Walsh': 0.0,
            'C10_H0': 0.0,
            'C11_H1': 0.0,
            'C12_Beta': c12
        }, index=df.index)
        
        ticker_features[ticker] = feat_df

    # 2. Compute Geometric Features (Heavy Computation)
    common_index = data.index
    asset_list = [t for t in tickers if t in ticker_features]
    n_days = len(common_index)
    n_assets = len(asset_list)
    
    # Create Returns Matrix (Time, Assets)
    returns_matrix = np.zeros((n_days, n_assets))
    for i, ticker in enumerate(asset_list):
        returns_matrix[:, i] = ticker_features[ticker]['C1_LogRet'].values
        
    lookback = 64
    walsh_lookback = 32
    
    print("Computing Geometric Features (Laplacian & Topology)...")
    
    # Pre-allocate
    laplacian_matrix = np.zeros((n_days, n_assets))
    h0_vec = np.zeros(n_days)
    h1_vec = np.zeros(n_days)
    
    # Parallelize the time loop for Laplacian & TDA
    if parallel:
        # Note: joblib overhead can be high for small functions, but TDA/Laplacian is heavy enough.
        # We process from t=lookback to n_days
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_time_step)(t, lookback, asset_list, returns_matrix, walsh_lookback)
            for t in tqdm(range(lookback, n_days))
        )
        
        # Collect results
        for res in results:
            t = res['t']
            laplacian_matrix[t, :] = res['residuals']
            h0_vec[t] = res['h0']
            h1_vec[t] = res['h1']
            
    else:
        # Sequential
        for t in tqdm(range(lookback, n_days)):
            window = returns_matrix[t-lookback:t, :].T
            laplacian_matrix[t, :] = get_laplacian_residuals(window, threshold=0.5)
            topo = get_tda_features(window)
            h0_vec[t] = topo[0]
            h1_vec[t] = topo[1]

    # 3. Walsh Score (Requires History of Residuals)
    # This must be done AFTER Laplacian is computed for the whole timeline (or at least causally).
    # Since we computed Laplacian for all T above, we can now compute Walsh.
    
    print("Computing Walsh Scores...")
    walsh_matrix = np.zeros((n_days, n_assets))
    
    # This loop is (Time * Assets). 
    # Can parallelize by Asset since they are independent given the Residual Matrix.
    
    def process_asset_walsh(i):
        w_col = np.zeros(n_days)
        # Vectorized rolling window might be possible but tricky with custom func.
        # Loop t:
        # Optimization: Only start after we have enough residuals
        start_t = lookback + walsh_lookback
        for t in range(start_t, n_days):
            res_hist = laplacian_matrix[t-walsh_lookback+1:t+1, i]
            w_col[t] = get_walsh_score(res_hist)
        return i, w_col

    # Parallelize by Asset
    if parallel:
        walsh_results = Parallel(n_jobs=n_jobs)(
            delayed(process_asset_walsh)(i) for i in range(n_assets)
        )
        for i, col in walsh_results:
            walsh_matrix[:, i] = col
    else:
        for i in range(n_assets):
            _, col = process_asset_walsh(i)
            walsh_matrix[:, i] = col

    # Fill back into DataFrames
    print("Merging features...")
    for i, ticker in enumerate(asset_list):
        ticker_features[ticker]['C8_Laplacian'] = laplacian_matrix[:, i]
        ticker_features[ticker]['C9_Walsh'] = walsh_matrix[:, i]
        ticker_features[ticker]['C10_H0'] = h0_vec
        ticker_features[ticker]['C11_H1'] = h1_vec
        
        # Fill NaNs
        ticker_features[ticker].fillna(0, inplace=True)
        
    return ticker_features, common_index

def scale_walsh_start(lookback, walsh):
    return lookback + walsh

def create_dataset(ticker_features, tickers, window_len=64):
    """
    Create (N, 12, 64) tensors for TCN.
    """
    all_X = []
    all_y = [] 
    
    print("Creating tensors...")
    
    for ticker in tqdm(tickers):
        if ticker not in ticker_features:
            continue
        
        df = ticker_features[ticker]
        data_values = df.values 
        
        # Targets: Next day return > 0
        targets = (df['C1_LogRet'].shift(-1) > 0).astype(int).values
        
        n_samples = len(data_values) - window_len - 1
        
        # Can we vectorize this slicing? 
        # Creating a strided view is efficient but complex for 3D tensor output.
        # List append is okay for reasonable sizes.
        
        for i in range(n_samples):
            x_window = data_values[i : i + window_len].T
            y_label = float(targets[i + window_len - 1])
            all_X.append(x_window)
            all_y.append(y_label)
            
    return np.array(all_X), np.array(all_y)
