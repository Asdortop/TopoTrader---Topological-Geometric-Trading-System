import pandas as pd

ASSET_CLASSES = {
    "Tech (FAANG)": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"],
    "Financials": ["JPM", "BAC", "GS", "V", "MA", "BRK-B", "CAT", "IBM"],
    "Healthcare": ["UNH", "JNJ", "PFE", "MRK", "ABBV", "AMGN"],
    "Energy": ["XOM", "CVX", "USO", "UNG"],
    "Indices (ETFs)": ["SPY", "QQQ", "IWM", "DIA", "EEM"],
}
BASELINE = {
    "Tech (FAANG)": (1.45, 18.2),
    "Financials": (1.15, 15.8),
    "Healthcare": (1.28, 16.4),
    "Energy": (0.92, 12.1),
    "Indices (ETFs)": (1.35, 16.9),
    "Overall": (1.30, 17.9),
}

v1 = pd.read_csv("asset_evaluation/v1_backtest_2022.csv", index_col=0)
v2 = pd.read_csv("asset_evaluation/v2_backtest_2022.csv", index_col=0)
wf = pd.read_csv("reports/walk_forward_us.csv")

print("=== WALK-FORWARD OOS (most reliable metric) ===")
print(wf[["window", "regime_label", "hit_rate", "confident_trade_pct"]].to_string(index=False))
print(f"Mean hit rate: {wf['hit_rate'].mean():.1%}")

rows = []
for cls, tickers in ASSET_CLASSES.items():
    b_sh, b_ret = BASELINE[cls]
    s1 = v1[v1.index.isin(tickers)]["sharpe"].mean()
    s2 = v2[v2.index.isin(tickers)]["sharpe"].mean()
    r1 = v1[v1.index.isin(tickers)]["annual_return"].mean() * 100
    r2 = v2[v2.index.isin(tickers)]["annual_return"].mean() * 100
    rows.append({
        "Asset Class": cls,
        "Baseline Sharpe": b_sh,
        "V1 Sharpe": round(s1, 2),
        "V2 Sharpe": round(s2, 2),
        "V2 - V1": round(s2 - s1, 2),
        "V1 Return %": round(r1, 1),
        "V2 Return %": round(r2, 1),
    })

b_sh, _ = BASELINE["Overall"]
rows.append({
    "Asset Class": "Overall",
    "Baseline Sharpe": b_sh,
    "V1 Sharpe": round(v1["sharpe"].mean(), 2),
    "V2 Sharpe": round(v2["sharpe"].mean(), 2),
    "V2 - V1": round(v2["sharpe"].mean() - v1["sharpe"].mean(), 2),
    "V1 Return %": round(v1["annual_return"].mean() * 100, 1),
    "V2 Return %": round(v2["annual_return"].mean() * 100, 1),
})

summary = pd.DataFrame(rows)
summary.to_csv("asset_evaluation/comparison_summary.csv", index=False)
print("\n=== ASSET CLASS COMPARISON (2022 backtest) ===")
print(summary.to_string(index=False))
