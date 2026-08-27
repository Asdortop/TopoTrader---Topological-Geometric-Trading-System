"""
Compare V1 vs V2 backtest results by asset class.
Uses the same evaluation logic as run_asset_evaluation.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from topo_trader.utils.data_loader import (
    fetch_universe_tickers,
    fetch_and_prepare_data,
    generate_features,
)
from topo_trader.models.tcn import MarketTCN

ASSET_CLASSES = {
    "Tech (FAANG)": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"],
    "Financials": ["JPM", "BAC", "GS", "V", "MA", "BRK-B", "CAT", "IBM"],
    "Healthcare": ["UNH", "JNJ", "PFE", "MRK", "ABBV", "AMGN"],
    "Energy": ["XOM", "CVX", "USO", "UNG"],
    "Indices (ETFs)": ["SPY", "QQQ", "IWM", "DIA", "EEM"],
}

BASELINE_V1 = {
    "Tech (FAANG)": {"count": 8, "avg_sharpe": 1.45, "avg_return": 18.2, "best": "NVDA (2.1)"},
    "Financials": {"count": 8, "avg_sharpe": 1.15, "avg_return": 15.8, "best": "JPM (1.8)"},
    "Healthcare": {"count": 6, "avg_sharpe": 1.28, "avg_return": 16.4, "best": "UNH (1.9)"},
    "Energy": {"count": 4, "avg_sharpe": 0.92, "avg_return": 12.1, "best": "XOM (1.2)"},
    "Indices (ETFs)": {"count": 5, "avg_sharpe": 1.35, "avg_return": 16.9, "best": "SPY (1.6)"},
    "Overall": {"count": 59, "avg_sharpe": 1.30, "avg_return": 17.9, "best": "-"},
}


def compute_metrics(returns: np.ndarray, trading_days: int = 252) -> dict:
    """Metrics for daily log-return series (signal * log_ret)."""
    if len(returns) == 0:
        return {"annual_return": 0.0, "sharpe": 0.0}
    mean_daily = float(np.mean(returns))
    std_daily = float(np.std(returns))
    # Correct annualization for log returns
    annual_return = float(np.expm1(mean_daily * trading_days))
    sharpe = (mean_daily / std_daily) * np.sqrt(trading_days) if std_daily > 0 else 0.0
    return {"annual_return": annual_return, "sharpe": sharpe}


def evaluate_model_on_period(
    features,
    tickers,
    model,
    device,
    start_date,
    end_date,
    num_inputs=16,
    window_len=64,
):
    """Mirror run_asset_evaluation.py, restricted to a date range."""
    results = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    for ticker in tickers:
        if ticker not in features:
            continue

        df = features[ticker]
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if len(df) <= window_len + 1:
            continue

        data_values = df.values.astype(np.float32)
        if data_values.shape[1] > num_inputs:
            data_values = data_values[:, :num_inputs]

        inputs = []
        for i in range(len(data_values) - window_len):
            inputs.append(data_values[i : i + window_len].T)
        inputs = np.array(inputs, dtype=np.float32)

        X = torch.tensor(inputs, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = model(X).flatten().cpu().numpy()

        signals = np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0)).astype(float)
        aligned_returns = df["C1_LogRet"].iloc[window_len:-1].values

        n = min(len(signals), len(aligned_returns))
        strategy_returns = signals[:n] * aligned_returns[:n]
        metrics = compute_metrics(strategy_returns)
        results.append({"ticker": ticker, **metrics})

    return pd.DataFrame(results).set_index("ticker")


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls, tickers in ASSET_CLASSES.items():
        sub = df[df.index.isin(tickers)]
        if sub.empty:
            continue
        best = sub.sort_values("sharpe", ascending=False).iloc[0]
        rows.append({
            "Asset Class": cls,
            "Count": len(sub),
            "Avg Sharpe": round(sub["sharpe"].mean(), 2),
            "Avg Return": round(sub["annual_return"].mean() * 100, 1),
            "Best Asset": f"{best.name} ({best['sharpe']:.1f})",
        })
    rows.append({
        "Asset Class": "Overall",
        "Count": len(df),
        "Avg Sharpe": round(df["sharpe"].mean(), 2),
        "Avg Return": round(df["annual_return"].mean() * 100, 1),
        "Best Asset": "-",
    })
    return pd.DataFrame(rows)


def load_model(version: str, device):
    if version == "v1":
        model = MarketTCN(num_inputs=12, num_channels=[32, 32, 32, 32])
        path = os.path.join("topo_trader", "models", "tcn_full.pth")
    else:
        model = MarketTCN(num_inputs=16, num_channels=[32, 32, 32, 32])
        path = os.path.join("topo_trader", "models", "tcn_v2_us.pth")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_or_build_features(tickers, version: str, cache_path=None):
    """Cache features per version (V1 Laplacian 12ch, V2 GAT 16ch)."""
    import pickle
    if cache_path is None:
        cache_path = f"topo_trader/data/cache/backtest_features_{version}_meta.pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached {version.upper()} features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data = fetch_and_prepare_data(tickers, start_date="2015-01-01", end_date="2024-01-01")
    features, _ = generate_features(data, tickers, parallel=False, n_jobs=1, version=version)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Features cached to {cache_path}")
    return features


def main():
    # Cache ends 2022-12-30; evaluate full 2022 (matches most of "2022-2023" label)
    start_date = "2022-01-01"
    end_date = "2022-12-30"

    print(f"=== Backtest Comparison ({start_date} -> {end_date}) ===")
    print("Note: cached data ends 2022-12-30, so 2023 is not available yet.\n")

    tickers = fetch_universe_tickers()
    features_v1 = load_or_build_features(tickers, "v1")
    features_v2 = load_or_build_features(tickers, "v2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_v1 = evaluate_model_on_period(
        features_v1, tickers, load_model("v1", device), device,
        start_date, end_date, num_inputs=12,
    )
    df_v2 = evaluate_model_on_period(
        features_v2, tickers, load_model("v2", device), device,
        start_date, end_date, num_inputs=16,
    )

    sum_v1 = summarize_by_class(df_v1)
    sum_v2 = summarize_by_class(df_v2)

    print("--- Previous V1 Baseline (from your table, 2022-2023) ---")
    for cls, v in BASELINE_V1.items():
        print(f"  {cls:20s}  Sharpe {v['avg_sharpe']:.2f}  Return {v['avg_return']:.1f}%")

    print("\n--- V1 Re-run (tcn_full.pth, 12ch, 2022 only) ---")
    print(sum_v1.to_string(index=False))

    print("\n--- V2 New (tcn_v2_us.pth, 16ch, 2022 only) ---")
    print(sum_v2.to_string(index=False))

    print("\n--- Delta: V2 - Previous Baseline ---")
    for cls in list(ASSET_CLASSES.keys()) + ["Overall"]:
        r2 = sum_v2[sum_v2["Asset Class"] == cls]
        if r2.empty or cls not in BASELINE_V1:
            continue
        ds = r2.iloc[0]["Avg Sharpe"] - BASELINE_V1[cls]["avg_sharpe"]
        dr = r2.iloc[0]["Avg Return"] - BASELINE_V1[cls]["avg_return"]
        print(f"  {cls:20s}  Sharpe {ds:+.2f}  Return {dr:+.1f}pp")

    print("\n--- Delta: V2 - V1 Re-run (same period, same methodology) ---")
    for cls in ASSET_CLASSES:
        r1 = sum_v1[sum_v1["Asset Class"] == cls]
        r2 = sum_v2[sum_v2["Asset Class"] == cls]
        if r1.empty or r2.empty:
            continue
        ds = r2.iloc[0]["Avg Sharpe"] - r1.iloc[0]["Avg Sharpe"]
        dr = r2.iloc[0]["Avg Return"] - r1.iloc[0]["Avg Return"]
        print(f"  {cls:20s}  Sharpe {ds:+.2f}  Return {dr:+.1f}pp")

    r1o = sum_v1[sum_v1["Asset Class"] == "Overall"].iloc[0]
    r2o = sum_v2[sum_v2["Asset Class"] == "Overall"].iloc[0]
    print(f"  {'Overall':20s}  Sharpe {r2o['Avg Sharpe'] - r1o['Avg Sharpe']:+.2f}  Return {r2o['Avg Return'] - r1o['Avg Return']:+.1f}pp")

    out_dir = os.path.dirname(__file__)
    df_v1.to_csv(os.path.join(out_dir, "v1_backtest_2022.csv"))
    df_v2.to_csv(os.path.join(out_dir, "v2_backtest_2022.csv"))
    sum_v2.to_csv(os.path.join(out_dir, "v2_summary_by_class.csv"), index=False)


if __name__ == "__main__":
    main()
