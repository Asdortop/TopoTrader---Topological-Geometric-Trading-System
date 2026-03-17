import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import os
from topo_trader.utils.data_loader import fetch_and_prepare_data, generate_features, fetch_universe_tickers
from topo_trader.models.tcn import MarketTCN

# Set Style
plt.style.use("dark_background")
sns.set_palette("bright")


def get_data_slice(data, target_date, lookback=64):
    """Get window ending at target_date."""
    idx = data.index.get_loc(target_date) if target_date in data.index else data.index.searchsorted(target_date)
    
    if isinstance(idx, slice):
        idx = idx.start
        
    start_idx = max(0, idx - lookback)
    return data.iloc[start_idx:idx]

def plot_market_graph(data, target_date, threshold=0.5, save_path="market_graph.png"):
    """
    Reconstruct and plot the market graph G=(V, E) for a specific date.
    """
    print(f"Generating Market Graph for {target_date}...")
    
    # 1. Get return window
    window_data = get_data_slice(data, target_date, lookback=64)
    if len(window_data) < 10:
        print("Not enough data for graph.")
        return

    # Extract log returns for all tickers
    tickers = data.columns.levels[0]
    returns_dict = {}
    for ticker in tickers:
        try:
            # Simple log ret
            closes = window_data[ticker]['Close']
            ret = np.log(closes / closes.shift(1)).dropna()
            returns_dict[ticker] = ret
        except:
            pass
            
    df_ret = pd.DataFrame(returns_dict).fillna(0)
    
    # 2. Compute Correlation & Adjacency
    corr = df_ret.corr().values
    tickers_list = df_ret.columns.tolist()
    
    # Adjacency: Hard Thresholding
    adj = np.where(np.abs(corr) > threshold, 1, 0)
    np.fill_diagonal(adj, 0)
    
    # 3. NetworkX Plot
    G = nx.Graph()
    for i, ticker in enumerate(tickers_list):
        G.add_node(ticker)
        
    # Add Edges
    rows, cols = np.where(adj == 1)
    for r, c in zip(rows, cols):
        if r < c: # Undirected
            weight = corr[r, c]
            G.add_edge(tickers_list[r], tickers_list[c], weight=weight)
            
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='#00ffcc', alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#ffffff')
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', font_family='sans-serif')
    
    plt.title(f"Market Topology: {target_date}\nThreshold={threshold} (Correlations)", fontsize=16, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='black')
    print(f"Saved {save_path}")
    plt.close()

def plot_residual_heatmap(features, tickers, save_path="residual_heatmap.png"):
    """
    Plot Laplacian Residuals (Mispricing) across assets over time.
    """
    print("Generating Residual Heatmap...")
    
    # Aggregate Channel 8 (Laplacian)
    res_data = {}
    for t in tickers:
        if t in features:
            res_data[t] = features[t]['C8_Laplacian']
            
    df_res = pd.DataFrame(res_data)
    
    # Take last 200 days for visibility
    df_plot = df_res.tail(100).T # Assets x Time
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(df_plot, cmap="vlag", center=0, robust=True, cbar_kws={'label': 'Laplacian Residual'})
    plt.title("Locomis Heatmap (Laplacian Mispricing)", fontsize=16)
    plt.xlabel("Time")
    plt.ylabel("Asset")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()


def _plot_return_distributions(strategy_returns, benchmark_returns, save_path):
    """
    Plot side-by-side return distributions for strategy vs benchmark.
    """
    plt.figure(figsize=(8, 5))
    sns.kdeplot(strategy_returns, label="Geometric Strategy", color="#00ffcc", shade=True)
    sns.kdeplot(benchmark_returns, label="Buy & Hold", color="gray", shade=True)
    plt.title("Daily Return Distribution: Strategy vs Buy & Hold")
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()


def _plot_confusion_matrix(pred_signals, realized_returns, save_path):
    """
    Confusion matrix for directional prediction (Up / Down).
    """
    # Map to {0,1} where 1 = Up, 0 = Down
    y_true = (realized_returns > 0).astype(int)

    # Map signals: +1 -> Up, -1 -> Down, 0 -> Neutral (ignored)
    mask = pred_signals != 0
    if mask.sum() == 0:
        print("No trading signals for confusion matrix.")
        return

    y_true = y_true[mask]
    y_pred = np.where(pred_signals[mask] > 0, 1, 0)

    # 2x2 confusion matrix
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Down", "Pred Up"],
        yticklabels=["True Down", "True Up"],
    )
    plt.title("Directional Confusion Matrix (Signals Where We Trade)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()


def plot_residual_vs_return(features, ticker, start_date=None, end_date=None, save_path="residual_vs_return.png"):
    """
    Scatter of Laplacian residual vs next-day return for a given asset.
    """
    if ticker not in features:
        return

    df = features[ticker].copy()
    if start_date is not None or end_date is not None:
        df = df.loc[start_date:end_date]

    x = df["C8_Laplacian"]
    y = df["C1_LogRet"].shift(-1)
    mask = (~x.isna()) & (~y.isna())
    x = x[mask]
    y = y[mask]

    # Ensure directory exists
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, alpha=0.4, s=20)
    plt.axhline(0, color="white", linestyle="--", alpha=0.5)
    plt.axvline(0, color="white", linestyle="--", alpha=0.5)
    plt.xlabel("Laplacian Residual (Locomis)")
    plt.ylabel("Next-Day Log Return")
    plt.title(f"Laplacian Residual vs Next-Day Return: {ticker}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()


def _compute_signals_and_returns(df, model, window_len=64):
    """
    Build rolling windows for one ticker and compute signals + aligned next-day returns.
    """
    data_values = df.values
    if len(df) <= window_len + 1:
        return None, None

    windows = []
    realized = []
    for i in range(len(df) - window_len - 1):
        windows.append(data_values[i : i + window_len].T)
        realized.append(df["C1_LogRet"].iloc[i + window_len])

    inputs = np.array(windows)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        probs = model(inputs_tensor).flatten().numpy()

    signals = np.zeros_like(probs)
    signals[probs > 0.55] = 1
    signals[probs < 0.45] = -1

    realized = np.array(realized)

    min_len = min(len(signals), len(realized))
    return signals[:min_len], realized[:min_len]


def plot_performance_for_ticker(features, model, ticker, start_date=None, end_date=None, out_dir="."):
    """
    For a single ticker, plot equity curve, return distribution,
    and confusion matrix. Outputs are saved into out_dir.
    """
    if ticker not in features:
        return

    df = features[ticker].copy()
    if start_date is not None or end_date is not None:
        df = df.loc[start_date:end_date]

    signals, aligned_returns = _compute_signals_and_returns(df, model, window_len=64)
    if signals is None or len(signals) == 0:
        print(f"Not enough data for performance plots for {ticker}.")
        return

    strategy_returns = signals * aligned_returns

    cumulative_strategy = np.cumsum(strategy_returns)
    cumulative_bh = np.cumsum(aligned_returns)

    os.makedirs(out_dir, exist_ok=True)

    # Equity curve
    equity_path = os.path.join(out_dir, "equity_curve.png")
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_strategy, label="Geometric Strategy", color="#00ffcc")
    plt.plot(cumulative_bh, label=f"Buy & Hold ({ticker})", color="gray", linestyle="--")
    plt.title(f"Performance Verification: {ticker}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(equity_path, dpi=300)
    print(f"Saved {equity_path}")
    plt.close()

    # Return distributions
    dist_path = os.path.join(out_dir, "returns_distribution.png")
    _plot_return_distributions(strategy_returns, aligned_returns, dist_path)

    # Confusion matrix (only for days we trade)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    _plot_confusion_matrix(signals, aligned_returns, cm_path)


def run_visualization():
    # Training period (for the model, handled elsewhere): 2015-01-01 to 2023-12-31
    # Here we focus on visualization for the test period 2024-01-01 to 2025-12-31.
    test_start = "2024-01-01"
    test_end = "2025-12-31"

    # 1. Load Data (include both train and test years so topology plots still work)
    tickers = fetch_universe_tickers()
    data = fetch_and_prepare_data(
        tickers, start_date="2015-01-01", end_date="2026-01-01"
    )

    # 2. Plot Topology (Crash vs Normal) - still using 2020/2021 examples
    plot_market_graph(data, "2020-03-20", threshold=0.6, save_path="market_graph_crash.png")
    plot_market_graph(data, "2021-01-15", threshold=0.6, save_path="market_graph_normal.png")

    # 3. Generate Features for all tickers
    features, _ = generate_features(data, tickers, parallel=True)

    # Optional heatmap on all tickers (or subset if this is too dense)
    plot_residual_heatmap(features, tickers)

    # 4. Load Model (assumed trained on 2015-2023)
    model = MarketTCN(num_inputs=12, num_channels=[32, 32, 32, 32])
    model_path = "topo_trader/models/tcn_full.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print("Loaded trained model.")
    else:
        print("Warning: No trained model found. Using random weights.")

    # 5. Per-asset visualizations for test period
    base_dir = os.path.join("outputs", "visualizations")
    os.makedirs(base_dir, exist_ok=True)

    for ticker in tickers:
        ticker_dir = os.path.join(base_dir, ticker)

        # Equity curve, return distribution, confusion matrix
        plot_performance_for_ticker(
            features, model, ticker, start_date=test_start, end_date=test_end, out_dir=ticker_dir
        )

        # Residual vs return
        residual_path = os.path.join(ticker_dir, "residual_vs_return.png")
        plot_residual_vs_return(
            features, ticker, start_date=test_start, end_date=test_end, save_path=residual_path
        )

if __name__ == "__main__":
    run_visualization()
