"""
TopoTrader V3 — India-Specific Sector Map (Week 10)
====================================================
Static sector groupings for all 49 NSE Nifty-50 tickers.

Used in:
  - Sector-aware hybrid graph (gat_engine_v2.py)
  - Sector-level analysis and reporting

Sectors based on NSE GICS classification as of 2021.
"""

# ── Sector definitions ─────────────────────────────────────────────────────────

SECTOR_MAP = {
    # Banking & Finance (NBFC + PSU + Private)
    "HDFCBANK":     "Banking",
    "ICICIBANK":    "Banking",
    "KOTAKBANK":    "Banking",
    "SBIN":         "Banking",
    "AXISBANK":     "Banking",
    "INDUSINDBK":   "Banking",
    "BAJFINANCE":   "NBFC",
    "BAJAJFINSV":   "NBFC",
    "HDFC":         "NBFC",

    # IT / Technology
    "TCS":          "IT",
    "INFY":         "IT",
    "WIPRO":        "IT",
    "HCLTECH":      "IT",
    "TECHM":        "IT",

    # FMCG / Consumer
    "HINDUNILVR":   "FMCG",
    "ITC":          "FMCG",
    "NESTLEIND":    "FMCG",
    "BRITANNIA":    "FMCG",
    "ASIANPAINT":   "FMCG",
    "TITAN":        "FMCG",

    # Pharma / Healthcare
    "SUNPHARMA":    "Pharma",
    "DRREDDY":      "Pharma",
    "CIPLA":        "Pharma",
    "DIVISLAB":     "Pharma",

    # Auto / Mobility
    "MARUTI":       "Auto",
    "TATAMOTORS":   "Auto",
    "BAJAJ-AUTO":   "Auto",
    "HEROMOTOCO":   "Auto",
    "EICHERMOT":    "Auto",

    # Energy / Oil & Gas
    "RELIANCE":     "Energy",
    "ONGC":         "Energy",
    "BPCL":         "Energy",
    "IOC":          "Energy",
    "POWERGRID":    "Energy",
    "NTPC":         "Energy",
    "GAIL":         "Energy",

    # Metals / Materials
    "TATASTEEL":    "Metals",
    "JSWSTEEL":     "Metals",
    "HINDALCO":     "Metals",
    "COALINDIA":    "Metals",
    "VEDL":         "Metals",

    # Telecom
    "BHARTIARTL":   "Telecom",

    # Infrastructure / Cement
    "ULTRACEMCO":   "Infra",
    "SHREECEM":     "Infra",
    "ADANIPORTS":   "Infra",
    "L&T":          "Infra",
    "LT":           "Infra",

    # Diversified / Conglomerate
    "M&M":          "Auto",
    "MM":           "Auto",
}

# ── Sector adjacency matrix weight (within-sector boost) ──────────────────────
# Stocks in the same sector get an additional edge weight.
# This prevents crashes from making all stocks look identical.

SECTOR_WEIGHT_BOOST = 0.30   # 30% of edge weight from sector membership

def get_sector(ticker: str) -> str:
    """Return sector for a given ticker. Falls back to 'Other'."""
    # Handle common name variations
    t = ticker.upper().replace('.NS', '').replace('.BO', '')
    return SECTOR_MAP.get(t, "Other")

def same_sector(ticker_a: str, ticker_b: str) -> bool:
    """True if two tickers belong to the same sector."""
    return get_sector(ticker_a) == get_sector(ticker_b)

def sector_adjacency_matrix(tickers: list) -> "np.ndarray":
    """
    Build a static binary sector adjacency matrix.

    Returns:
        (N, N) matrix: 1.0 if same sector, 0.0 otherwise
    """
    import numpy as np
    n = len(tickers)
    adj = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if i != j and same_sector(a, b):
                adj[i, j] = 1.0
    return adj

def get_all_sectors(tickers: list) -> dict:
    """Return {ticker: sector} dict for a list of tickers."""
    return {t: get_sector(t) for t in tickers}

def print_sector_summary(tickers: list):
    """Print grouped sector overview for the given ticker list."""
    from collections import defaultdict
    groups = defaultdict(list)
    for t in tickers:
        groups[get_sector(t)].append(t)
    print("Sector composition:")
    for sector, stocks in sorted(groups.items()):
        print(f"  {sector:<15}: {', '.join(stocks)}")


if __name__ == '__main__':
    # Quick test
    test_tickers = list(SECTOR_MAP.keys())
    print_sector_summary(test_tickers)
    m = sector_adjacency_matrix(test_tickers)
    import numpy as np
    print(f"\nSector adjacency matrix: {m.shape}, "
          f"non-zero entries: {int(m.sum())}")
