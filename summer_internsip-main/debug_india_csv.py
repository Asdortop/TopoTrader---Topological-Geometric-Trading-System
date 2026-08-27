"""Step-by-step India CSV debug to find exact crash line."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd

csv_path = "topo_trader/data/india_raw/ADANIPORTS.csv"

print("Step 1: read_csv", flush=True)
df = pd.read_csv(csv_path)
print(f"  shape={df.shape}  dtypes={df.dtypes['Open']}", flush=True)

print("Step 2: to_datetime", flush=True)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
print(f"  Date sample: {df['Date'].iloc[0]}", flush=True)

print("Step 3: set_index", flush=True)
df = df.sort_values("Date").set_index("Date")

print("Step 4: select columns", flush=True)
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
print(f"  dtypes: {df.dtypes.to_dict()}", flush=True)

print("Step 5: date filter", flush=True)
df = df.loc["2015-01-01":"2022-12-31"]
print(f"  rows after filter: {len(df)}", flush=True)

print("Step 6: astype float", flush=True)
print(f"  Volume sample: {df['Volume'].head(3).tolist()}", flush=True)
df = df.astype(float)
print(f"  SUCCESS! Shape: {df.shape}", flush=True)
