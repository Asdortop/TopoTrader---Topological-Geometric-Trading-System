"""
extend_india_nse_direct.py
==========================
Downloads NSE Bhavcopy data directly via NSE's public REST API
using requests with session management — no yfinance, no jugaad-data threading.

Same approach as jugaad-data internally but:
  - No concurrent.futures (no C-level thread crash)
  - No file caching (no Windows path issues)
  - Simple, robust, transparent

Extends all 49 Kaggle Nifty-50 CSVs from ~2021 to 2025.
Run once: python extend_india_nse_direct.py
"""

import csv
import os
import sys
import time
import json
import traceback
from datetime import datetime, date, timedelta

sys.stdout.reconfigure(line_buffering=True)

import requests

# ── Config ─────────────────────────────────────────────────────────────────────
INDIA_CSV_DIR = "topo_trader/data/india_raw"
CACHE_FILE    = "topo_trader/data/cache/india_csv_data.parquet"
FETCH_START   = date(2022, 1, 1)
FETCH_END     = date(2025, 6, 30)   # extend to mid-2025

CSV_COLUMNS = [
    "Date", "Symbol", "Series", "Prev Close", "Open", "High", "Low",
    "Last", "Close", "VWAP", "Volume", "Turnover", "Trades",
    "Deliverable Volume", "%Deliverble"
]

SKIP_TICKERS = {"HDFC", "INFRATEL"}   # merged/delisted

SYMBOL_MAP = {"MM": "M&M"}            # Kaggle name -> NSE symbol

# NSE request chunk size (API limited to ~365 days per request)
CHUNK_DAYS = 90

# ── NSE Session ────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent"      : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept"          : "application/json, text/plain, */*",
    "Accept-Language" : "en-US,en;q=0.9",
    "Referer"         : "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}

NSE_BASE = "https://www.nseindia.com"
NSE_API  = (
    "https://www.nseindia.com/api/historical/cm/equity"
    "?symbol={symbol}&series=[%22EQ%22]&from={from_date}&to={to_date}"
)


def make_session() -> requests.Session:
    """Create an authenticated NSE session by hitting the homepage first."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        # Warm up session (NSE requires cookies from homepage visit)
        r = session.get(NSE_BASE, timeout=15)
        r.raise_for_status()
        time.sleep(1)
    except Exception as e:
        print(f"  [WARN] NSE homepage warmup failed: {e}", flush=True)
    return session


def fetch_nse_chunk(session: requests.Session, symbol: str,
                    from_d: date, to_d: date) -> list:
    """
    Fetch one chunk of data for symbol between from_d and to_d.
    Returns list of dicts matching rohanrao CSV format.
    """
    url = NSE_API.format(
        symbol    = requests.utils.quote(symbol),
        from_date = from_d.strftime("%d-%m-%Y"),
        to_date   = to_d.strftime("%d-%m-%Y"),
    )
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception as e:
        print(f"    [WARN] Request failed for {symbol} {from_d}-{to_d}: {e}", flush=True)
        return []

    records = data.get("data", [])
    if not records:
        return []

    rows = []
    for rec in records:
        # NSE API response field names
        row_date = rec.get("CH_TIMESTAMP", rec.get("TIMESTAMP", ""))
        # Parse date (format: YYYY-MM-DD or DD-MMM-YYYY)
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y"):
            try:
                parsed = datetime.strptime(str(row_date).strip()[:10], fmt)
                row_date = parsed.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue
        else:
            continue  # skip unparseable dates

        rows.append({
            "Date"              : row_date,
            "Symbol"            : symbol,
            "Series"            : rec.get("CH_SERIES", "EQ"),
            "Prev Close"        : rec.get("CH_PREVIOUS_CLS_PRICE", ""),
            "Open"              : rec.get("CH_OPENING_PRICE", ""),
            "High"              : rec.get("CH_TRADE_HIGH_PRICE", ""),
            "Low"               : rec.get("CH_TRADE_LOW_PRICE", ""),
            "Last"              : rec.get("CH_LAST_TRADED_PRICE", ""),
            "Close"             : rec.get("CH_CLOSING_PRICE", ""),
            "VWAP"              : rec.get("VWAP", ""),
            "Volume"            : rec.get("CH_TOT_TRADED_QTY", ""),
            "Turnover"          : rec.get("CH_TOT_TRADED_VAL", ""),
            "Trades"            : rec.get("CH_TOTAL_TRADES", ""),
            "Deliverable Volume": rec.get("COP_DELIV_QTY", ""),
            "%Deliverble"       : rec.get("COP_DELIV_PERC", ""),
        })
    return rows


def fetch_all_chunks(session: requests.Session, symbol: str,
                     from_d: date, to_d: date) -> list:
    """Fetch data in CHUNK_DAYS chunks, combining results."""
    all_rows = []
    current  = from_d
    while current < to_d:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), to_d)
        chunk = fetch_nse_chunk(session, symbol, current, chunk_end)
        all_rows.extend(chunk)
        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)   # rate limiting
    return all_rows


# ── Native CSV helpers (zero pandas) ──────────────────────────────────────────

def get_max_date_native(csv_path: str):
    """Return max date in CSV using Python's csv module. Zero pandas."""
    max_date = None
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("Date", "").strip()
                for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y"):
                    try:
                        d = datetime.strptime(raw, fmt).date()
                        if max_date is None or d > max_date:
                            max_date = d
                        break
                    except ValueError:
                        continue
    except Exception as e:
        print(f"    [WARN] Could not read {csv_path}: {e}", flush=True)
    return max_date


def get_existing_dates(csv_path: str) -> set:
    """Return set of date strings already in the CSV."""
    dates = set()
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row.get("Date", "").strip()
                if d:
                    dates.add(d)
    except Exception:
        pass
    return dates


def append_rows(csv_path: str, new_rows: list, existing_dates: set) -> int:
    """Append deduplicated rows to CSV. Returns count added."""
    to_write = sorted(
        [r for r in new_rows if r["Date"] not in existing_dates],
        key=lambda r: r["Date"]
    )
    if not to_write:
        return 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        for row in to_write:
            writer.writerow(row)
    return len(to_write)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    skip_files = {"NIFTY50_all", "stock_metadata", "INFRATEL"}
    tickers = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(INDIA_CSV_DIR)
        if f.endswith(".csv") and os.path.splitext(f)[0] not in skip_files
    )
    print(f"Found {len(tickers)} tickers.")
    print(f"Fetching NSE data: {FETCH_START} -> {FETCH_END}\n")

    # Warm up NSE session once for all tickers
    print("Establishing NSE session...", flush=True)
    session = make_session()
    print("Session ready.\n", flush=True)

    total_added = 0
    results = []

    for i, ticker in enumerate(tickers):
        csv_path = os.path.join(INDIA_CSV_DIR, f"{ticker}.csv")
        nse_sym  = SYMBOL_MAP.get(ticker, ticker)
        label    = f"[{i+1:2d}/{len(tickers)}] {ticker:20s}"

        if ticker in SKIP_TICKERS:
            print(f"  {label} SKIP (delisted/merged)", flush=True)
            results.append((ticker, "SKIP", 0))
            continue

        max_date = get_max_date_native(csv_path)
        if max_date and max_date >= FETCH_END:
            print(f"  {label} already complete ({max_date})", flush=True)
            results.append((ticker, "DONE", 0))
            continue

        fetch_from = FETCH_START
        if max_date and max_date >= FETCH_START:
            fetch_from = max_date + timedelta(days=1)

        print(f"  {label} {fetch_from} -> {FETCH_END} ...", end="", flush=True)

        try:
            new_rows = fetch_all_chunks(session, nse_sym, fetch_from, FETCH_END)
        except BaseException as e:
            print(f" FAIL: {e}", flush=True)
            results.append((ticker, "ERROR", 0))
            # Refresh session on error
            session = make_session()
            continue

        if not new_rows:
            print(f" NO DATA", flush=True)
            results.append((ticker, "NO_DATA", 0))
            continue

        existing = get_existing_dates(csv_path)
        n_added  = append_rows(csv_path, new_rows, existing)
        total_added += n_added
        drange = f"{new_rows[0]['Date']} -> {new_rows[-1]['Date']}"
        print(f" +{n_added} rows [{drange}]", flush=True)
        results.append((ticker, "OK", n_added))

        time.sleep(1.0)   # polite delay between tickers

    # ── Summary ────────────────────────────────────────────────────────────────
    ok      = [r for r in results if r[1] == "OK"]
    no_data = [r for r in results if r[1] == "NO_DATA"]
    errors  = [r for r in results if r[1] == "ERROR"]

    print(f"\n{'='*65}")
    print(f"  Extended       : {len(ok)} tickers  (+{total_added:,} rows)")
    print(f"  No data        : {len(no_data)} -> {[r[0] for r in no_data]}")
    print(f"  Errors         : {len(errors)} -> {[r[0] for r in errors]}")

    if len(ok) > 0:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print(f"\n  Parquet cache deleted — pipeline will rebuild from full 2004-2025 data.")
        print(f"\n  Next step: python baseline_comparison.py --market india")

    # Verify
    if ok:
        sample = ok[0][0]
        sp     = os.path.join(INDIA_CSV_DIR, f"{sample}.csv")
        dates  = get_existing_dates(sp)
        if dates:
            sd = sorted(dates)
            print(f"\n  Verification ({sample}): {sd[0]} -> {sd[-1]}  ({len(sd):,} rows)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
    except Exception as e:
        print(f"\nFATAL: {e}", flush=True)
        traceback.print_exc()
