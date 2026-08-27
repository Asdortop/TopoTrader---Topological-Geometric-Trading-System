"""Quick smoke test for India CSV loader."""
from topo_trader.utils.data_loader import fetch_india_csv_tickers, load_india_csv_data

tickers = fetch_india_csv_tickers()
print(f"Found {len(tickers)} tickers: {tickers[:5]}")

data = load_india_csv_data(start_date="2015-01-01", end_date="2022-12-31", force_reload=True)
print(f"Shape: {data.shape}")

t0 = list(data.columns.get_level_values(0).unique())[0]
print(f"First ticker: {t0}")
print(f"Date range: {data.index[0].date()} -> {data.index[-1].date()}")
print(f"Sample Close (last 3 days):")
print(data[t0]["Close"].tail(3))
print("\nIndia CSV loader OK!")
