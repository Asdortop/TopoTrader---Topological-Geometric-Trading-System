"""
extend_india_jugaad.py
======================
Extends rohanrao Kaggle Nifty-50 CSVs (2004-2021) forward to 2025
using jugaad-data's NSE Bhavcopy pull.

Both data sources are NSE Bhavcopy -> identical format, zero conversion.

Design:
  - Uses Python's native csv module (NOT pandas) for all file I/O
    to avoid the C-level pandas crash on Windows.
  - Downloads per-ticker via jugaad_data.nse.stock_df()
  - Appends new rows in rohanrao column order
  - Idempotent: safe to re-run, deduplicates by date
  - Deletes parquet cache on success so pipeline sees full data

Run:
    python extend_india_jugaad.py
"""

import csv
import os
import sys
import time
import traceback
from datetime import datetime, date, timedelta

# Force line-buffered stdout so every print appears immediately
sys.stdout.reconfigure(line_buffering=True)

# ── Fix jugaad-data Windows cache bug ─────────────────────────────────────────
# jugaad-data creates cache filenames from datetime.__str__() which includes
# colons (e.g. "2022-03-01 00:00:00") — colons are ILLEGAL in Windows filenames.
# Fix: redirect cache to a safe dir AND monkeypatch kw_to_fname to strip colons.
_JUGAAD_CACHE = os.path.join(os.getcwd(), "topo_trader", "data", "cache", "jugaad_cache")
os.makedirs(_JUGAAD_CACHE, exist_ok=True)
os.environ["J_CACHE_DIR"] = _JUGAAD_CACHE

import jugaad_data.util as _jutil
_original_kw_to_fname = _jutil.kw_to_fname

def _safe_kw_to_fname(**kw):
    name = _original_kw_to_fname(**kw)
    # Replace all characters illegal on Windows
    for ch in [':', '*', '?', '<', '>', '|', '"', '\\', '/']:
        name = name.replace(ch, '-')
    return name

_jutil.kw_to_fname = _safe_kw_to_fname
# ── End jugaad-data fix ────────────────────────────────────────────────────────

# ── Config ─────────────────────────────────────────────────────────────────────
INDIA_CSV_DIR = "topo_trader/data/india_raw"
CACHE_FILE    = "topo_trader/data/cache/india_csv_data.parquet"
FETCH_START   = date(2022, 1, 1)   # start of extension window
FETCH_END     = date(2025, 7, 31)  # extend as far forward as possible

# Rohanrao CSV column order (must match exactly for append)
CSV_COLUMNS = [
    "Date", "Symbol", "Series", "Prev Close", "Open", "High", "Low",
    "Last", "Close", "VWAP", "Volume", "Turnover", "Trades",
    "Deliverable Volume", "%Deliverble"
]

# Tickers that were delisted / merged and have no post-2021 data
SKIP_TICKERS = {
    "HDFC",     # Merged into HDFCBANK on 1-Jul-2023 (no full year available)
    "INFRATEL", # Already excluded from pipeline
}

# Kaggle CSV name -> NSE jugaad symbol (only overrides needed)
SYMBOL_MAP = {
    "MM": "M&M",  # Kaggle saved M&M as MM to avoid shell issues
}


# ==============================================================================
# STEP 1: Find max date in an existing CSV using native Python csv reader
# ==============================================================================

def get_max_date_native(csv_path: str):
    """
    Read max date from CSV using Python's built-in csv module.
    Zero pandas -> zero C-level crash risk.
    Returns datetime.date or None.
    """
    max_date = None
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("Date", "").strip()
                if not raw:
                    continue
                try:
                    # Handle both YYYY-MM-DD and DD-MM-YYYY formats
                    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y"):
                        try:
                            d = datetime.strptime(raw, fmt).date()
                            if max_date is None or d > max_date:
                                max_date = d
                            break
                        except ValueError:
                            continue
                except Exception:
                    continue
    except Exception as e:
        print(f"    [WARN] Could not read {csv_path}: {e}", flush=True)
    return max_date


# ==============================================================================
# STEP 2: Download via jugaad-data
# ==============================================================================

def download_jugaad(symbol: str, from_date: date, to_date: date):
    """
    Download NSE Bhavcopy data via jugaad-data.
    Returns list of dicts with rohanrao column names, or empty list.
    """
    from jugaad_data.nse import stock_df as jstock_df

    try:
        df = jstock_df(
            symbol=symbol,
            from_date=datetime.combine(from_date, datetime.min.time()),
            to_date=datetime.combine(to_date, datetime.min.time()),
            series="EQ"
        )
    except Exception as e:
        print(f"    [WARN] jugaad-data error for {symbol}: {e}", flush=True)
        return []

    if df is None or len(df) == 0:
        return []

    # Detect which date column jugaad returned
    date_col = None
    for c in ["CH_TIMESTAMP", "DATE", "Date", "TIMESTAMP"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        print(f"    [WARN] No date column found in jugaad output. Cols: {df.columns.tolist()}", flush=True)
        return []

    # Column mapping: jugaad -> rohanrao
    col_map = {
        "CH_OPENING_PRICE"    : "Open",
        "CH_TRADE_HIGH_PRICE" : "High",
        "CH_TRADE_LOW_PRICE"  : "Low",
        "CH_LAST_TRADED_PRICE": "Last",
        "CH_CLOSING_PRICE"    : "Close",
        "CH_PREVIOUS_CLS_PRICE": "Prev Close",
        "VWAP"                : "VWAP",
        "CH_TOT_TRADED_QTY"   : "Volume",
        "CH_TOT_TRADED_VAL"   : "Turnover",
        "CH_TOTAL_TRADES"     : "Trades",
        "CH_SERIES"           : "Series",
        # Alternate names some jugaad versions use
        "OPEN"                : "Open",
        "HIGH"                : "High",
        "LOW"                 : "Low",
        "LAST"                : "Last",
        "CLOSE"               : "Close",
        "PREV_CLOSE"          : "Prev Close",
        "TOTAL_TRADED_QTY"    : "Volume",
        "TOTAL_TRADED_VALUE"  : "Turnover",
        "TOTAL_TRADES"        : "Trades",
    }

    rows = []
    for _, r in df.iterrows():
        # Parse date
        raw_date = r[date_col]
        try:
            if hasattr(raw_date, "date"):
                row_date = raw_date.date() if callable(raw_date.date) else raw_date.date
            else:
                for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y"):
                    try:
                        row_date = datetime.strptime(str(raw_date).strip()[:10], fmt).date()
                        break
                    except ValueError:
                        continue
        except Exception:
            continue

        row = {
            "Date"              : row_date.strftime("%Y-%m-%d"),
            "Symbol"            : symbol,
            "Series"            : "EQ",
            "Prev Close"        : "",
            "Open"              : "",
            "High"              : "",
            "Low"               : "",
            "Last"              : "",
            "Close"             : "",
            "VWAP"              : "",
            "Volume"            : "",
            "Turnover"          : "",
            "Trades"            : "",
            "Deliverable Volume": "",
            "%Deliverble"       : "",
        }

        for jugaad_col, rohan_col in col_map.items():
            if jugaad_col in r.index and str(r[jugaad_col]) not in ("nan", "None", ""):
                row[rohan_col] = str(r[jugaad_col])

        rows.append(row)

    return rows


# ==============================================================================
# STEP 3: Append rows to CSV using native Python csv writer
# ==============================================================================

def append_rows_native(csv_path: str, new_rows: list, existing_dates: set) -> int:
    """
    Append new_rows to csv_path, skipping dates already in existing_dates.
    Uses Python's native csv module — zero pandas.
    Returns number of rows actually appended.
    """
    to_write = [r for r in new_rows if r["Date"] not in existing_dates]
    if not to_write:
        return 0

    # Sort by date before appending
    to_write.sort(key=lambda r: r["Date"])

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        for row in to_write:
            writer.writerow(row)

    return len(to_write)


def get_existing_dates_native(csv_path: str) -> set:
    """Return set of all date strings already in the CSV."""
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


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Collect tickers
    skip_files = {"NIFTY50_all", "stock_metadata", "INFRATEL"}
    tickers = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(INDIA_CSV_DIR)
        if f.endswith(".csv") and os.path.splitext(f)[0] not in skip_files
    )
    print(f"Found {len(tickers)} tickers.")
    print(f"Extending {FETCH_START} -> {FETCH_END} via jugaad-data NSE Bhavcopy\n")
    print(f"{'='*65}")

    total_added = 0
    results     = []

    for i, ticker in enumerate(tickers):
        csv_path = os.path.join(INDIA_CSV_DIR, f"{ticker}.csv")
        nse_sym  = SYMBOL_MAP.get(ticker, ticker)
        label    = f"[{i+1:2d}/{len(tickers)}] {ticker:20s}"

        # Skip delisted tickers
        if ticker in SKIP_TICKERS:
            print(f"  {label} SKIP (delisted/merged)", flush=True)
            results.append((ticker, "SKIP", 0))
            continue

        # Find max date already in CSV (native reader, no pandas)
        max_date = get_max_date_native(csv_path)
        if max_date and max_date >= FETCH_END:
            print(f"  {label} already complete to {max_date}", flush=True)
            results.append((ticker, "ALREADY_COMPLETE", 0))
            continue

        fetch_from = FETCH_START
        if max_date and max_date >= FETCH_START:
            fetch_from = max_date + timedelta(days=1)

        print(f"  {label} fetching {fetch_from} -> {FETCH_END} ...", flush=True, end="")

        try:
            new_rows = download_jugaad(nse_sym, fetch_from, FETCH_END)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            traceback.print_exc()
            results.append((ticker, f"DOWNLOAD_ERROR", 0))
            time.sleep(1)
            continue

        if not new_rows:
            print(f" NO DATA returned", flush=True)
            results.append((ticker, "NO_DATA", 0))
            time.sleep(1)
            continue

        # Get existing dates to deduplicate
        existing_dates = get_existing_dates_native(csv_path)

        try:
            n_added = append_rows_native(csv_path, new_rows, existing_dates)
        except Exception as e:
            print(f" APPEND_ERROR: {e}", flush=True)
            results.append((ticker, "APPEND_ERROR", 0))
            continue

        total_added += n_added
        date_range = f"{new_rows[0]['Date']} -> {new_rows[-1]['Date']}"
        print(f" +{n_added} rows  [{date_range}]", flush=True)
        results.append((ticker, "OK", n_added))

        # Polite rate limiting to avoid NSE blocking
        time.sleep(1.2)

    # ── Summary ────────────────────────────────────────────────────────────────
    ok      = [r for r in results if r[1] == "OK"]
    skipped = [r for r in results if r[1] in ("SKIP", "ALREADY_COMPLETE")]
    failed  = [r for r in results if r[1] not in ("OK", "SKIP", "ALREADY_COMPLETE")]

    print(f"\n{'='*65}")
    print(f"  Extended         : {len(ok)} tickers  (+{total_added:,} total rows)")
    print(f"  Skipped          : {len(skipped)} tickers")
    print(f"  Failed/No data   : {len(failed)} tickers")
    if failed:
        print(f"  Failed tickers   : {[r[0] for r in failed]}")

    # ── Delete cache so pipeline rebuilds from full 2004-2025 data ─────────────
    if len(ok) > 0:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print(f"\n  Cache deleted: {CACHE_FILE}")
        else:
            print(f"\n  Cache not found (will be built fresh on next run)")
        print("  Next step: python baseline_comparison.py --market india")
    else:
        print("\n  No tickers extended — cache NOT deleted.")

    # ── Verify one ticker ───────────────────────────────────────────────────────
    if len(ok) > 0:
        sample_ticker = ok[0][0]
        sample_path   = os.path.join(INDIA_CSV_DIR, f"{sample_ticker}.csv")
        dates_sample  = get_existing_dates_native(sample_path)
        if dates_sample:
            sorted_d = sorted(dates_sample)
            print(f"\n  Verification ({sample_ticker}):")
            print(f"    First date : {sorted_d[0]}")
            print(f"    Last date  : {sorted_d[-1]}")
            print(f"    Total rows : {len(dates_sample):,}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", flush=True)
        traceback.print_exc()
