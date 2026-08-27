"""Quick step-by-step debug for extend_india_data.py."""
import sys, os, pandas as pd, yfinance as yf
sys.stdout.reconfigure(line_buffering=True)

CSV_PATH = "topo_trader/data/india_raw/ADANIPORTS.csv"
YF_TICKER = "ADANIPORTS.NS"

print("Step 1: read CSV", flush=True)
df = pd.read_csv(CSV_PATH)
print(f"  shape={df.shape}  cols={df.columns.tolist()[:4]}", flush=True)

print("Step 2: parse dates", flush=True)
dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce").dropna()
max_date = dates.max()
print(f"  max_date={max_date.date()}", flush=True)

print("Step 3: yfinance download", flush=True)
obj  = yf.Ticker(YF_TICKER)
hist = obj.history(start="2022-01-01", end="2023-12-31", auto_adjust=True, raise_errors=False)
print(f"  hist shape={hist.shape}", flush=True)
print(f"  hist cols={hist.columns.tolist()}", flush=True)
print(f"  hist head:\n{hist.head(2)}", flush=True)

print("Step 4: convert to Kaggle format", flush=True)
hist2 = hist.reset_index()
hist2["Date"] = pd.to_datetime(hist2["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
print(f"  date sample: {hist2['Date'].iloc[0]}", flush=True)

print("Step 5: build new rows", flush=True)
new_rows = pd.DataFrame({
    "Date": hist2["Date"], "Symbol": "ADANIPORTS", "Series": "EQ",
    "Prev Close": hist2["Close"].shift(1), "Open": hist2["Open"],
    "High": hist2["High"], "Low": hist2["Low"], "Last": hist2["Close"],
    "Close": hist2["Close"], "VWAP": float("nan"), "Volume": hist2["Volume"],
    "Turnover": float("nan"), "Trades": float("nan"),
    "Deliverable Volume": float("nan"), "%Deliverble": float("nan"),
})
print(f"  new_rows shape={new_rows.shape}", flush=True)

print("Step 6: append to CSV", flush=True)
new_rows.to_csv(CSV_PATH, mode="a", header=False, index=False)
print(f"  SUCCESS! Appended {len(new_rows)} rows", flush=True)
