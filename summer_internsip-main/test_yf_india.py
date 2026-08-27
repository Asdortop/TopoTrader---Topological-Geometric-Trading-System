"""Test yfinance availability for Indian NSE tickers."""
import sys
sys.stdout.reconfigure(line_buffering=True)
import yfinance as yf
import time

# Test sample of tickers across sectors
test_tickers = ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS",
                "TATAMOTORS.NS", "SBIN.NS", "ITC.NS", "WIPRO.NS"]

print(f"Testing {len(test_tickers)} tickers for 2022-2023 data availability...\n")

working = []
failed  = []

for ticker in test_tickers:
    try:
        hist = yf.download(ticker, start="2022-01-01", end="2023-12-31",
                           progress=False, auto_adjust=True)
        if len(hist) > 100:
            print(f"  OK   {ticker:20s}  rows={len(hist)}  "
                  f"last={hist.index[-1].date()}", flush=True)
            working.append(ticker)
        else:
            print(f"  FAIL {ticker:20s}  rows={len(hist)} (too few)", flush=True)
            failed.append(ticker)
    except Exception as e:
        print(f"  ERR  {ticker:20s}  {e}", flush=True)
        failed.append(ticker)
    time.sleep(1)

print(f"\nResult: {len(working)} working, {len(failed)} failed")
print(f"Working: {working}")
print(f"Failed:  {failed}")
