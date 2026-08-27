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
from ..strategies.gat_engine import get_gat_signal          # V2: replaces Laplacian
from ..strategies.topology_engine import get_tda_features
from ..strategies.walsh_filter import get_walsh_score

CACHE_DIR = "topo_trader/data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_universe_tickers():
    """US S&P 100 + ETF universe (V1 default)."""
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
        "JPM", "XOM", "V", "PG", "HD", "MA", "CVX", "ABBV", "PEP", "KO",
        "LLY", "BAC", "COST", "MRK", "AVGO", "TMO", "DIS", "PFE", "CSCO", "ACN", "SPY",
        "QQQ", "IWM", "DIA", "EEM", "TLT", "GLD", "SLV", "USO", "UNG",
        "AMD", "INTC", "QCOM", "TXN", "HON", "UNP", "LIN", "PM", "IBM", "AMGN",
        "CAT", "GS", "BA", "MMM", "GE", "RTX", "LMT", "DE",
    ]
    return list(set(tickers))


def fetch_india_tickers():
    """
    Nifty 200 universe -- Indian equities via yfinance (.NS suffix).
    Covers large-cap + mid-cap across all major NSE sectors:
    IT, Banking, FMCG, Auto, Pharma, Energy, Metals, Telecom, Cement, Real Estate.
    """
    tickers = [
        # -- Information Technology -----------------------------------------
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
        "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS",
        # -- Banking & Financial Services -----------------------------------
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
        "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS",
        # -- Consumer Goods / FMCG -----------------------------------------
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
        "DABUR.NS", "MARICO.NS", "COLPAL.NS", "GODREJCP.NS", "EMAMILTD.NS",
        # -- Automobiles ---------------------------------------------------
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
        "EICHERMOT.NS", "TVSMOTORS.NS", "ASHOKLEY.NS",
        # -- Pharmaceuticals -----------------------------------------------
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
        "BIOCON.NS", "TORNTPHARM.NS", "AUROPHARMA.NS", "ALKEM.NS",
        # -- Energy & Oil / Gas --------------------------------------------
        "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS",
        "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANITRANS.NS", "BPCL.NS", "IOC.NS",
        # -- Metals & Mining -----------------------------------------------
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS",
        "COALINDIA.NS", "NMDC.NS", "SAIL.NS",
        # -- Telecom -------------------------------------------------------
        "BHARTIARTL.NS",
        # -- Cement --------------------------------------------------------
        "ULTRACEMCO.NS", "SHREECEM.NS", "ACC.NS", "AMBUJACEM.NS", "JKCEMENT.NS",
        # -- Capital Goods / Engineering -----------------------------------
        "LT.NS", "SIEMENS.NS", "ABB.NS", "BHEL.NS", "THERMAX.NS",
        # -- Consumer Discretionary ----------------------------------------
        "TITAN.NS", "TRENT.NS", "DMART.NS", "NYKAA.NS", "JUBLFOOD.NS",
        # -- Real Estate ---------------------------------------------------
        "DLF.NS", "GODREJPROP.NS", "PRESTIGE.NS", "OBEROIRLTY.NS",
        # -- Market Proxy --------------------------------------------------
        "^NSEI",   # Nifty 50 index as market proxy (replaces SPY role)
    ]
    return list(set(tickers))


INDIA_CSV_DIR = "topo_trader/data/india_raw"


def fetch_india_csv_tickers():
    """
    Return ticker names derived from the Kaggle Nifty-50 CSV filenames.
    Excludes meta-files like NIFTY50_all.csv and stock_metadata.csv.
    """
    if not os.path.isdir(INDIA_CSV_DIR):
        raise FileNotFoundError(f"India CSV directory not found: {INDIA_CSV_DIR}")
    skip = {"NIFTY50_all.csv", "stock_metadata.csv", "INFRATEL.csv"}  # skip empty/meta files
    names = [
        os.path.splitext(f)[0]
        for f in sorted(os.listdir(INDIA_CSV_DIR))
        if f.endswith(".csv") and f not in skip
    ]
    return names


def load_india_csv_data(start_date="2015-01-01", end_date="2024-01-01",
                        cache_name="india_csv_data.parquet", force_reload=False):
    """
    Load Nifty-50 historical data from local Kaggle CSVs.

    Produces the same MultiIndex(ticker, field) DataFrame format as
    fetch_and_prepare_data() so generate_features() works unchanged.

    Columns per ticker: Open, High, Low, Close, Volume
    """
    cache_path = os.path.join(CACHE_DIR, cache_name)
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading India data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    tickers = fetch_india_csv_tickers()
    print(f"Loading {len(tickers)} India CSVs from {INDIA_CSV_DIR} ...")

    all_dfs = {}
    for ticker in tickers:
        csv_path = os.path.join(INDIA_CSV_DIR, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            # Read WITHOUT parse_dates to avoid slow date inference
            df = pd.read_csv(csv_path)
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
            df = df.dropna(subset=["Date"])               # drop any rows with bad dates
            df = df.sort_values("Date").set_index("Date")
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            # Use boolean mask — avoids silent crash in newer pandas with .loc[str:str]
            mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
            df   = df.loc[mask]
            if len(df) < 100:
                continue
            all_dfs[ticker] = df.astype(float)
        except Exception as e:
            print(f"  Warning: could not load {ticker}.csv -- {e}")

    if not all_dfs:
        raise RuntimeError("No India CSV data loaded. Check topo_trader/data/india_raw/")

    # Align on common date index
    common_index = None
    for df in all_dfs.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    common_index = common_index.sort_values()

    # Build MultiIndex DataFrame: columns = (ticker, field)  <- same as yf.download(group_by='ticker')
    frames = {}
    for ticker, df in all_dfs.items():
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            frames[(ticker, col)] = df.reindex(common_index)[col].values

    mindex = pd.MultiIndex.from_tuples(frames.keys())
    data = pd.DataFrame(frames, index=common_index, columns=mindex)
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    try:
        data.to_parquet(cache_path)
        print(f"India data cached: {cache_path}  ({len(all_dfs)} tickers, {len(common_index)} days)")
    except Exception as e:
        print(f"Warning: could not cache -- {e}")

    return data


def fetch_and_prepare_data(tickers, start_date="2015-01-01", end_date="2024-01-01",
                           force_reload=False, cache_name="raw_market_data.parquet"):
    cache_path = os.path.join(CACHE_DIR, cache_name)
    
    if not force_reload and os.path.exists(cache_path):
        print(f"Loading data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Downloading data for {len(tickers)} tickers (individual mode)...")

    import time
    all_dfs = {}
    failed  = []

    for i, ticker in enumerate(tickers):
        for attempt in range(3):           # Up to 3 retries per ticker
            try:
                t_obj = yf.Ticker(ticker)
                hist  = t_obj.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    raise_errors=False,
                )
                if len(hist) > 0:
                    all_dfs[ticker] = hist
                    break
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(1.0)
        else:
            failed.append(ticker)

        if (i + 1) % 10 == 0:
            print(f"  Downloaded {i+1}/{len(tickers)} tickers  ({len(failed)} failed so far)")
            time.sleep(0.3)  # Polite rate limiting

    if failed:
        print(f"  Warning: {len(failed)} tickers failed -- {failed[:5]}{'...' if len(failed) > 5 else ''}")

    if not all_dfs:
        raise RuntimeError("No data downloaded -- check your internet connection or ticker symbols.")

    # Build a flat MultiIndex DataFrame: (field, ticker) columns -- same structure as old bulk download
    common_index = None
    for df in all_dfs.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    tuples = []
    arrays = {}
    for ticker, df in all_dfs.items():
        df = df.reindex(common_index).ffill().bfill()
        for col in df.columns:
            key = (col, ticker)
            tuples.append(key)
            arrays[key] = df[col].values

    import numpy as np_
    mindex = pd.MultiIndex.from_tuples(tuples)
    data   = pd.DataFrame(arrays, index=common_index, columns=mindex)
    data.columns.names = [None, None]

    # Save to cache
    try:
        data.to_parquet(cache_path)
        print(f"Data cached to {cache_path}  ({len(all_dfs)} tickers, {len(common_index)} days)")
    except Exception as e:
        print(f"Warning: Could not cache data: {e}")

    return data

def process_time_step(t, lookback, asset_list, returns_matrix, walsh_lookback,
                       use_gat: bool = True):
    """
    Helper for parallel processing a single time step.

    V2 change: `use_gat=True` routes C8 through the attention-weighted GAT signal
    (adaptive threshold + softmax aggregation) instead of the V1 fixed Laplacian.

    Note on parallelism:
        Geometric features depend only on window [t-lookback : t] -- independent per t.
        Walsh score depends on the residual history, so it must run in a second pass.
    """
    window = returns_matrix[t-lookback:t, :].T  # (Assets, Lookback)

    # -- C8: Graph Signal (V2 GAT or V1 Laplacian) --------------------------
    if use_gat:
        residuals = get_gat_signal(window, percentile=70)
    else:
        residuals = get_laplacian_residuals(window, threshold=0.5)

    # -- C10, C11: Topology -------------------------------------------------
    topo = get_tda_features(window)

    return {
        't'        : t,
        'residuals': residuals,
        'h0'       : topo[0],
        'h1'       : topo[1],
    }

def generate_features(data, tickers, parallel=True, n_jobs=-1, version="v2"):
    """
    Build per-ticker feature DataFrames.

    version="v1": 12 channels (Laplacian C8, no regime channels) -- matches tcn_full.pth
    version="v2": 16 channels (GAT C8 + regime one-hot C13-C16) -- matches tcn_v2_us.pth
    """
    use_gat = version != "v1"
    n_channels = 16 if version == "v2" else 12
    print(f"Generating features ({version.upper()}, {n_channels} channels)...")
    
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
        
        # -- Feature DataFrame (12ch V1 or 16ch V2) -----------------------
        cols = {
            'C1_LogRet'  : c1,
            'C2_Vol'     : c2,
            'C3_RSI'     : c3,
            'C4_MACD'    : c4,
            'C5_ATR'     : c5,
            'C6_BB'      : c6,
            'C7_ZScore'  : c7,
            'C8_GAT'     : 0.0,
            'C9_Walsh'   : 0.0,
            'C10_H0'     : 0.0,
            'C11_H1'     : 0.0,
            'C12_Beta'   : c12,
        }
        if version == "v2":
            cols.update({
                'C13_Regime_Crash'   : 0.0,
                'C14_Regime_HighVol' : 0.0,
                'C15_Regime_Bull'    : 0.0,
                'C16_Regime_Sideways': 1.0,
            })
        feat_df = pd.DataFrame(cols, index=df.index)
        
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
    
    print("Computing Geometric Features (V2 GAT Signal & Topology)..." if use_gat
          else "Computing Geometric Features (V1 Laplacian Signal & Topology)...")

    # Pre-allocate
    graph_matrix = np.zeros((n_days, n_assets))
    h0_vec     = np.zeros(n_days)
    h1_vec     = np.zeros(n_days)

    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_time_step)(t, lookback, asset_list, returns_matrix,
                                       walsh_lookback, use_gat=use_gat)
            for t in tqdm(range(lookback, n_days))
        )
        for res in results:
            t = res['t']
            graph_matrix[t, :] = res['residuals']
            h0_vec[t]          = res['h0']
            h1_vec[t]          = res['h1']
    else:
        for t in tqdm(range(lookback, n_days)):
            window = returns_matrix[t-lookback:t, :].T
            if use_gat:
                graph_matrix[t, :] = get_gat_signal(window, percentile=70)
            else:
                graph_matrix[t, :] = get_laplacian_residuals(window, threshold=0.5)
            topo      = get_tda_features(window)
            h0_vec[t] = topo[0]
            h1_vec[t] = topo[1]

    # -- V2 only: Regime Labels (C13-C16) ------------------------------------
    regime_crash = regime_highvol = regime_bull = regime_sideways = None
    if version == "v2":
        print("Computing Regime Labels (V2)...")
        mean_atr       = np.zeros(n_days)
        for i, ticker in enumerate(asset_list):
            mean_atr  += ticker_features[ticker]['C5_ATR'].values
        mean_atr      /= max(n_assets, 1)

        spy_ret        = spy_log_ret.values if hasattr(spy_log_ret, 'values') else spy_log_ret
        spy_rolling60  = np.convolve(spy_ret, np.ones(60)/60, mode='same')

        nonzero_atr    = mean_atr[mean_atr > 0]
        atr_hi_thresh  = np.percentile(nonzero_atr, 75) if len(nonzero_atr) > 0 else 0.02

        CRASH_H0_THRESH = 1.5
        regime_crash    = np.zeros(n_days)
        regime_highvol  = np.zeros(n_days)
        regime_bull     = np.zeros(n_days)
        regime_sideways = np.ones(n_days)

        for t in range(lookback, n_days):
            h0  = h0_vec[t]
            atr = mean_atr[t]
            r60 = spy_rolling60[t] if t < len(spy_rolling60) else 0.0

            if h0 < CRASH_H0_THRESH and atr > atr_hi_thresh:
                regime_crash[t]    = 1.0
                regime_sideways[t] = 0.0
            elif atr > atr_hi_thresh:
                regime_highvol[t]  = 1.0
                regime_sideways[t] = 0.0
            elif r60 > 0:
                regime_bull[t]     = 1.0
                regime_sideways[t] = 0.0

    # 3. Walsh Score (Requires History of Residuals)
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
            res_hist = graph_matrix[t-walsh_lookback+1:t+1, i]
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

    merge_label = f"{version.upper()}: {n_channels} channels"
    print(f"Merging features ({merge_label})...")
    for i, ticker in enumerate(asset_list):
        df = ticker_features[ticker]
        df['C8_GAT']   = graph_matrix[:, i]
        df['C9_Walsh'] = walsh_matrix[:, i]
        df['C10_H0']   = h0_vec
        df['C11_H1']   = h1_vec
        if version == "v2":
            df['C13_Regime_Crash']    = regime_crash
            df['C14_Regime_HighVol']  = regime_highvol
            df['C15_Regime_Bull']     = regime_bull
            df['C16_Regime_Sideways'] = regime_sideways
        df.fillna(0, inplace=True)

    return ticker_features, common_index

def scale_walsh_start(lookback, walsh):
    return lookback + walsh

def create_dataset(ticker_features, tickers, window_len=64):
    """
    Create (N, n_channels, window_len) tensors for TCN.
    V2: n_channels = 16 (was 12 in V1).
    """
    all_X, all_y = [], []
    print("Creating tensors (V2: 16 channels)...")

    for ticker in tqdm(tickers):
        if ticker not in ticker_features:
            continue

        df          = ticker_features[ticker]
        data_values = df.values                                   # (n_days, 16)
        targets     = (df['C1_LogRet'].shift(-1) > 0).astype(int).values
        n_samples   = len(data_values) - window_len - 1

        if n_samples <= 0:
            continue

        for i in range(n_samples):
            x_window = data_values[i : i + window_len].T         # (16, window_len)
            y_label  = float(targets[i + window_len - 1])
            all_X.append(x_window)
            all_y.append(y_label)

    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)
