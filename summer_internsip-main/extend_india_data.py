"""
extend_india_data.py
====================
Extends existing Kaggle Nifty-50 CSVs (2004-2021) with yfinance data
for 2022-01-01 to 2023-12-31.

Strategy:
  - Reads all 49 tickers from topo_trader/data/india_raw/
  - Downloads each as TICKER.NS from yfinance
  - Appends rows in Kaggle-compatible CSV format (same columns)
  - Deletes the parquet cache so it gets rebuilt from full data
  - Skips tickers where the CSV already has 2022+ data (idempotent)

Run once:
    python extend_india_data.py
"""

import os
import time
import pandas as pd
import yfinance as yf

# ── Config ─────────────────────────────────────────────────────────────────────
INDIA_CSV_DIR = "topo_trader/data/india_raw"
CACHE_FILE    = "topo_trader/data/cache/india_csv_data.parquet"
START_DATE    = "2022-01-01"
END_DATE      = "2023-12-31"

# Known name mismatches between Kaggle filename and yfinance ticker
TICKER_MAP = {
    "MM"         : "M&M",
    "BAJAJ-AUTO" : "BAJAJ-AUTO",
    "HDFC"       : "HDFC",
    "ZEEL"       : "ZEEL",
}

# Delisted / merged tickers — skip gracefully (no 2022-2023 data available)
SKIP_TICKERS = {
    "HDFC",    # Merged into HDFCBANK on July 1 2023
}


def kaggle_ticker_to_yf(ticker: str) -> str:
    """Convert Kaggle CSV filename ticker to yfinance .NS format."""
    # Reverse the MM -> M&M mapping
    yf_base = TICKER_MAP.get(ticker, ticker)
    return f"{yf_base}.NS"


def get_csv_max_date(csv_path: str) -> pd.Timestamp:
    """Return the latest date in an existing CSV file."""
    # Read WITHOUT parse_dates to avoid C-level pandas crash on date columns
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        return pd.NaT
    dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    dates = dates.dropna()
    if len(dates) == 0:
        return pd.NaT
    return dates.max()


def yf_to_kaggle_rows(hist: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Convert yfinance history DataFrame to Kaggle CSV format.
    Kaggle columns: Date, Symbol, Series, Prev Close, Open, High, Low,
                    Last, Close, VWAP, Volume, Turnover, Trades,
                    Deliverable Volume, %Deliverble
    We only have OHLCV from yfinance; fill the rest with NaN.
    The loader only uses Open, High, Low, Close, Volume so NaN is fine.
    """
    if hist.empty:
        return pd.DataFrame()

    hist = hist.reset_index()
    # yfinance returns Date as datetime64 (possibly with tz-info)
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

    rows = pd.DataFrame({
        "Date"                : hist["Date"],
        "Symbol"              : symbol,
        "Series"              : "EQ",
        "Prev Close"          : hist["Close"].shift(1),
        "Open"                : hist["Open"],
        "High"                : hist["High"],
        "Low"                 : hist["Low"],
        "Last"                : hist["Close"],
        "Close"               : hist["Close"],
        "VWAP"                : float("nan"),
        "Volume"              : hist["Volume"],
        "Turnover"            : float("nan"),
        "Trades"              : float("nan"),
        "Deliverable Volume"  : float("nan"),
        "%Deliverble"         : float("nan"),
    })
    return rows


def extend_ticker(ticker: str, yf_ticker: str) -> dict:
    """
    Extend one ticker's CSV with yfinance data.
    Returns a status dict.
    """
    csv_path = os.path.join(INDIA_CSV_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        return {"ticker": ticker, "status": "MISSING_CSV", "rows_added": 0}

    # Skip known delisted / merged tickers
    if ticker in SKIP_TICKERS:
        return {"ticker": ticker, "status": "DELISTED_SKIP", "rows_added": 0}

    # Check what data we already have
    max_date = get_csv_max_date(csv_path)
    if pd.isna(max_date):
        fetch_start = START_DATE
    elif max_date >= pd.Timestamp("2023-12-01"):
        return {"ticker": ticker, "status": "ALREADY_COMPLETE", "rows_added": 0}
    else:
        # Fetch from day after last available date
        fetch_start = (max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Download from yfinance
    try:
        obj  = yf.Ticker(yf_ticker)
        hist = obj.history(start=fetch_start, end=END_DATE,
                           auto_adjust=True, raise_errors=False)
    except Exception as e:
        return {"ticker": ticker, "status": f"YF_ERROR: {e}", "rows_added": 0}

    if hist is None or len(hist) == 0:
        return {"ticker": ticker, "status": "NO_DATA", "rows_added": 0}

    # Convert to Kaggle format
    new_rows = yf_to_kaggle_rows(hist, symbol=ticker)
    if new_rows.empty:
        return {"ticker": ticker, "status": "EMPTY_AFTER_CONVERT", "rows_added": 0}

    # Append to CSV
    new_rows.to_csv(csv_path, mode="a", header=False, index=False)
    return {"ticker": ticker, "status": "OK", "rows_added": len(new_rows),
            "date_range": f"{new_rows['Date'].iloc[0]} -> {new_rows['Date'].iloc[-1]}"}


def main():
    skip = {"NIFTY50_all", "stock_metadata", "INFRATEL"}
    tickers = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(INDIA_CSV_DIR)
        if f.endswith(".csv") and os.path.splitext(f)[0] not in skip
    )
    print(f"Found {len(tickers)} tickers to extend.")
    print(f"Fetching {START_DATE} -> {END_DATE} from yfinance...\n")

    results = []
    failed  = []

    for i, ticker in enumerate(tickers):
        yf_ticker = kaggle_ticker_to_yf(ticker)
        try:
            result = extend_ticker(ticker, yf_ticker)
        except Exception as e:
            result = {"ticker": ticker, "status": f"EXCEPTION: {e}", "rows_added": 0}
        results.append(result)

        status = result["status"]
        rows   = result.get("rows_added", 0)
        drange = result.get("date_range", "")

        if status == "OK":
            print(f"  [{i+1:2d}/{len(tickers)}] {ticker:20s} +{rows:4d} rows  {drange}", flush=True)
        elif status == "ALREADY_COMPLETE":
            print(f"  [{i+1:2d}/{len(tickers)}] {ticker:20s} already complete", flush=True)
        elif status in ("NO_DATA", "DELISTED_SKIP"):
            print(f"  [{i+1:2d}/{len(tickers)}] {ticker:20s} SKIP: {status}", flush=True)
            failed.append(ticker)
        else:
            print(f"  [{i+1:2d}/{len(tickers)}] {ticker:20s} WARN: {status}", flush=True)
            failed.append(ticker)

        time.sleep(0.4)

    # ── Summary ─────────────────────────────────────────────────────────────
    ok_count   = sum(1 for r in results if r["status"] == "OK")
    skip_count = sum(1 for r in results if r["status"] == "ALREADY_COMPLETE")
    fail_count = len(failed)

    print(f"\n{'='*55}")
    print(f"  Extended : {ok_count} tickers")
    print(f"  Skipped  : {skip_count} (already had 2022-2023 data)")
    print(f"  Failed   : {fail_count}")
    if failed:
        print(f"  Failed tickers: {failed}")
    print(f"{'='*55}")

    # ── Delete cache so it gets rebuilt with new data ────────────────────────
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"\nCache deleted: {CACHE_FILE}")
        print("Next pipeline run will rebuild from complete 2004-2023 data.")
    else:
        print(f"\nNo cache file found at {CACHE_FILE} (will be created on next run).")

    # Verify one ticker to confirm the extension worked
    print("\nVerification -- checking TCS date range after extension:")
    df_verify = pd.read_csv(os.path.join(INDIA_CSV_DIR, "TCS.csv"), usecols=["Date"])
    df_verify["Date"] = pd.to_datetime(df_verify["Date"], errors="coerce")
    print(f"  TCS data: {df_verify['Date'].min().date()} -> {df_verify['Date'].max().date()}")
    print(f"  Total rows: {len(df_verify):,}")
    print("\nDone! Now run: python run_training_v2.py --market india")


if __name__ == "__main__":
    main()
