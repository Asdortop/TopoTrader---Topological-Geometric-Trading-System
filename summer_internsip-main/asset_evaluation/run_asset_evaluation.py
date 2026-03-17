import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Allow running this script from inside the 'asset_evaluation' folder
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from topo_trader.utils.data_loader import (
    fetch_universe_tickers,
    fetch_and_prepare_data,
    generate_features,
)
from topo_trader.models.tcn import MarketTCN


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_performance_metrics(returns: np.ndarray, trading_days: int = 252) -> dict:
    """
    Basic performance statistics for a daily return series.
    """
    if len(returns) == 0:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
        }

    mean_daily = float(np.mean(returns))
    std_daily = float(np.std(returns))

    annual_return = (1.0 + mean_daily) ** trading_days - 1.0
    annual_vol = std_daily * np.sqrt(trading_days)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    hit_rate = float(np.mean(returns > 0))

    return {
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
    }


def evaluate_all_assets(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    window_len: int = 64,
) -> pd.DataFrame:
    """
    Evaluate strategy and buy-and-hold performance for every asset
    in the universe over a fixed period.
    """
    print("=== Asset-wise Performance Evaluation ===")
    print(f"Period: {start_date} to {end_date}")

    # Make sure relative paths (cache, model path, etc.) behave like other scripts
    os.chdir(REPO_ROOT)

    tickers = fetch_universe_tickers()
    print(f"Universe size: {len(tickers)} tickers")

    # Load market data (cached if already downloaded)
    data = fetch_and_prepare_data(
        tickers,
        start_date=start_date,
        end_date=end_date,
    )

    # Generate features for all tickers
    features, _ = generate_features(data, tickers, parallel=True, n_jobs=-1)

    # Load trained model
    model = MarketTCN(num_inputs=12, num_channels=[32, 32, 32, 32])
    model_path = os.path.join("topo_trader", "models", "tcn_full.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Run 'run_training_full.py' first to create it."
        )

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    results = []

    for ticker in tickers:
        if ticker not in features:
            continue

        df = features[ticker]
        data_values = df.values

        if len(data_values) <= window_len + 1:
            # Not enough history for this ticker in the chosen period
            continue

        # Build rolling input windows (no shuffling)
        inputs = []
        for i in range(len(data_values) - window_len):
            inputs.append(data_values[i : i + window_len].T)

        inputs = np.array(inputs)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

        with torch.no_grad():
            probs = model(inputs_tensor).flatten().numpy()

        # Convert probabilities to trading signals (+1, -1, 0)
        signals = []
        for p in probs:
            if p > 0.55:
                signals.append(1)
            elif p < 0.45:
                signals.append(-1)
            else:
                signals.append(0)
        signals = np.array(signals, dtype=float)

        # Align with next-day returns of this ticker
        aligned_returns = df["C1_LogRet"].iloc[window_len:-1].values

        min_len = min(len(signals), len(aligned_returns))
        signals = signals[:min_len]
        aligned_returns = aligned_returns[:min_len]

        strategy_returns = signals * aligned_returns

        strat_metrics = compute_performance_metrics(strategy_returns)
        bh_metrics = compute_performance_metrics(aligned_returns)

        results.append(
            {
                "ticker": ticker,
                "strategy_annual_return": strat_metrics["annual_return"],
                "strategy_annual_volatility": strat_metrics["annual_volatility"],
                "strategy_sharpe": strat_metrics["sharpe"],
                "strategy_max_drawdown": strat_metrics["max_drawdown"],
                "strategy_hit_rate": strat_metrics["hit_rate"],
                "bh_annual_return": bh_metrics["annual_return"],
                "bh_annual_volatility": bh_metrics["annual_volatility"],
                "bh_sharpe": bh_metrics["sharpe"],
                "bh_max_drawdown": bh_metrics["max_drawdown"],
                "bh_hit_rate": bh_metrics["hit_rate"],
            }
        )

    df_results = pd.DataFrame(results).set_index("ticker").sort_values(
        by="strategy_sharpe", ascending=False
    )

    return df_results


def plot_all_assets(df_results: pd.DataFrame, output_dir: str) -> None:
    """
    Create visual summaries of per-asset performance.
    """
    if df_results.empty:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Use a consistent style
    plt.style.use("dark_background")
    sns.set_palette("bright")

    # 1) Bar plot of Sharpe ratios (top 25 assets)
    top_n = min(25, len(df_results))
    df_top = df_results.head(top_n).iloc[::-1]  # reverse for nicer horizontal plot

    plt.figure(figsize=(10, 0.4 * top_n + 2))
    sns.barplot(
        x="strategy_sharpe",
        y=df_top.index,
        data=df_top.reset_index(),
        color="#00ffcc",
    )
    plt.xlabel("Strategy Sharpe")
    plt.ylabel("Ticker")
    plt.title(f"Top {top_n} Assets by Strategy Sharpe")
    plt.tight_layout()
    sharpe_path = os.path.join(output_dir, "asset_sharpe_ranking.png")
    plt.savefig(sharpe_path, dpi=300)
    plt.close()

    # 2) Scatter: annual return vs Sharpe for all assets
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        df_results["strategy_sharpe"],
        df_results["strategy_annual_return"] * 100.0,
        c=df_results["strategy_hit_rate"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="none",
    )
    plt.colorbar(scatter, label="Hit Rate")
    plt.xlabel("Strategy Sharpe")
    plt.ylabel("Strategy Annual Return (%)")
    plt.title("Asset-wise Strategy Performance")
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "asset_performance_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()


def main() -> None:
    set_seed(42)

    df_results = evaluate_all_assets()

    # Save metrics CSV into this folder
    output_csv = os.path.join(THIS_DIR, "asset_performance_metrics.csv")
    df_results.to_csv(output_csv)
    print(f"\nSaved asset metrics to: {output_csv}")

    # Plots for quick visual comparison
    plot_all_assets(df_results, THIS_DIR)
    print("Saved aggregate asset performance plots in the same folder.")

    # Print top performers
    if not df_results.empty:
        best_by_sharpe = df_results.iloc[0]
        best_by_return = df_results.sort_values(
            by="strategy_annual_return", ascending=False
        ).iloc[0]

        print("\n=== Top Assets (Strategy) ===")
        print(f"Best by Sharpe: {best_by_sharpe.name}")
        print(f"  Sharpe: {best_by_sharpe['strategy_sharpe']:.2f}")
        print(f"  Annual return: {best_by_sharpe['strategy_annual_return']*100:.2f}%")

        print(f"\nBest by Annual Return: {best_by_return.name}")
        print(f"  Sharpe: {best_by_return['strategy_sharpe']:.2f}")
        print(f"  Annual return: {best_by_return['strategy_annual_return']*100:.2f}%")


if __name__ == "__main__":
    main()

