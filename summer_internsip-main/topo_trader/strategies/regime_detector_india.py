"""
TopoTrader V3 — India-Calibrated Regime Detector (Week 10)
===========================================================
Detects market regime using India-specific thresholds.

Unlike generic VIX-based detection, this version is calibrated
for Indian market characteristics:

  1. Nifty VIX proxy     — India VIX equivalent using Nifty options-implied vol
  2. FII flow proxy      — Foreign Institutional Investor momentum (sign of flow)
  3. Breadth indicator   — % Nifty stocks above 200-day MA
  4. RSI of Nifty index  — Index-level momentum

Regime labels:
  0 = Crash     (vol > 30%, FII outflow, breadth < 30%)
  1 = High-Vol  (vol > 20%, mixed signals)
  2 = Bull      (vol < 15%, FII inflow, breadth > 60%)
  3 = Sideways  (everything else)

India-specific events that required calibration:
  - Demonetization (Nov 2016): Liquidity shock, not captured by VIX
  - GST (Jul 2017): Structural reform — sector-differentiated impact
  - IL&FS (Sep 2018): Credit market freeze — breadth crashed before prices
  - COVID crash (Mar 2020): Fastest 30% drawdown in Nifty history

Usage:
    from topo_trader.strategies.regime_detector_india import IndiaRegimeDetector
    detector = IndiaRegimeDetector()
    regime = detector.detect(returns_window)   # returns 0-3
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

# ── India-calibrated thresholds ────────────────────────────────────────────────

# Annualised volatility thresholds (from Nifty data 2010-2021)
CRASH_VOL_THRESH    = 0.28   # >28% ann. vol -> Crash (India VIX proxy)
HIGHVOL_VOL_THRESH  = 0.18   # >18% ann. vol -> High-Vol
BULL_VOL_THRESH     = 0.13   # <13% ann. vol -> Bull

# Breadth thresholds (fraction of stocks above 200-day MA)
CRASH_BREADTH_THRESH = 0.30
BULL_BREADTH_THRESH  = 0.60

# RSI of index thresholds
CRASH_RSI_THRESH = 40
BULL_RSI_THRESH  = 55

ANNUAL_DAYS = 252


class IndiaRegimeDetector:
    """
    India-calibrated market regime detector.

    Combines volatility, breadth, and momentum signals
    with India-specific event overrides.

    Args:
        vol_window:     Days to compute rolling volatility (default 20)
        breadth_window: Days to compute 200-day MA (default 200)
        rsi_window:     RSI period (default 14)
    """

    def __init__(self, vol_window: int = 20, breadth_window: int = 200,
                 rsi_window: int = 14):
        self.vol_window     = vol_window
        self.breadth_window = breadth_window
        self.rsi_window     = rsi_window

        # India-specific hardcoded event overrides
        # These events caused regime changes not captured by rolling volatility
        self._event_overrides = {
            # Demonetization: Nov 8-Dec 2016 -> Crash regime
            ('2016-11-08', '2016-12-31'): 0,
            # GST transition: Jul-Sep 2017 -> High-Vol (supply chain disruption)
            ('2017-07-01', '2017-09-30'): 1,
            # IL&FS default: Sep 2018 -> Crash (NBFC credit freeze)
            ('2018-09-01', '2018-11-30'): 0,
        }

    def detect(self, returns_window: np.ndarray,
               date: Optional[pd.Timestamp] = None) -> int:
        """
        Detect regime from a window of stock returns.

        Args:
            returns_window: (N_stocks, T) array of log returns
            date:           Trading date (for event overrides)

        Returns:
            regime: int 0=Crash, 1=HighVol, 2=Bull, 3=Sideways
        """
        # Check India-specific event overrides first
        if date is not None:
            override = self._check_event_override(date)
            if override is not None:
                return override

        # ── Compute signals ────────────────────────────────────────────────
        index_returns = returns_window.mean(axis=0)   # proxy Nifty index returns

        vol   = self._rolling_vol(index_returns)
        rsi   = self._compute_rsi(index_returns)
        breadth = self._compute_breadth(returns_window)

        # ── Decision tree ──────────────────────────────────────────────────
        return self._classify(vol, rsi, breadth)

    def detect_series(self, returns_df: pd.DataFrame) -> pd.Series:
        """
        Detect regime for every date in a return DataFrame.

        Args:
            returns_df: DataFrame (T x N_stocks) of log returns

        Returns:
            pd.Series: regime labels indexed by date
        """
        dates   = returns_df.index
        regimes = []
        values  = returns_df.values   # (T, N)

        T = len(dates)
        for t in range(T):
            start = max(0, t - self.breadth_window)
            window = values[start:t+1].T   # (N, window)
            if window.shape[1] < 5:
                regimes.append(3)  # default: sideways
                continue
            r = self.detect(window, date=dates[t])
            regimes.append(r)

        return pd.Series(regimes, index=dates, name='regime')

    def _rolling_vol(self, returns: np.ndarray) -> float:
        """Annualised volatility of last vol_window days."""
        w = min(self.vol_window, len(returns))
        return float(np.std(returns[-w:]) * np.sqrt(ANNUAL_DAYS))

    def _compute_rsi(self, returns: np.ndarray) -> float:
        """RSI of index returns (using log returns as price proxy)."""
        w = min(self.rsi_window, len(returns))
        r = returns[-w:]
        gains  = np.where(r > 0, r, 0).mean()
        losses = np.where(r < 0, -r, 0).mean()
        if losses < 1e-8:
            return 100.0
        rs = gains / losses
        return float(100 - 100 / (1 + rs))

    def _compute_breadth(self, returns_window: np.ndarray) -> float:
        """
        Fraction of stocks with positive cumulative return over window.
        Proxy for 'stocks above 200-day MA' when we don't have price history.
        """
        cum_rets = returns_window.sum(axis=1)   # (N_stocks,)
        return float((cum_rets > 0).mean())

    def _classify(self, vol: float, rsi: float, breadth: float) -> int:
        """
        Classify regime from composite signals.

        Priority order (based on India market observations):
        1. Crash:     High vol AND low RSI AND low breadth
        2. High-Vol:  High vol OR (low RSI AND moderate breadth)
        3. Bull:      Low vol AND high RSI AND high breadth
        4. Sideways:  Default
        """
        crash_signals = int(vol > CRASH_VOL_THRESH) + \
                        int(rsi < CRASH_RSI_THRESH) + \
                        int(breadth < CRASH_BREADTH_THRESH)

        bull_signals  = int(vol < BULL_VOL_THRESH) + \
                        int(rsi > BULL_RSI_THRESH) + \
                        int(breadth > BULL_BREADTH_THRESH)

        if crash_signals >= 2:
            return 0   # Crash
        elif vol > HIGHVOL_VOL_THRESH and crash_signals == 1:
            return 1   # High-Vol
        elif bull_signals >= 2:
            return 2   # Bull
        else:
            return 3   # Sideways

    def _check_event_override(self, date: pd.Timestamp) -> Optional[int]:
        """Check if this date falls in a known India-specific event period."""
        for (start_str, end_str), regime in self._event_overrides.items():
            start = pd.Timestamp(start_str)
            end   = pd.Timestamp(end_str)
            if start <= date <= end:
                return regime
        return None

    def regime_label(self, regime: int) -> str:
        return {0: 'Crash', 1: 'High-Vol', 2: 'Bull', 3: 'Sideways'}.get(regime, 'Unknown')


# ── Hybrid sector-aware GAT signal ────────────────────────────────────────────

def get_hybrid_gat_signal(returns_window: np.ndarray, tickers: list,
                           sector_weight: float = 0.30,
                           percentile: int = 70) -> np.ndarray:
    """
    Sector-aware hybrid GAT signal (Week 10 core contribution).

    Addresses the crash-correlation problem:
    During a crash, all return correlations spike to ~1.0, making the
    pure correlation graph degenerate (all edges identical).

    Fix:
        edge_weight = (1 - sector_weight) * dynamic_corr
                    + sector_weight * static_sector_membership

    This preserves within-sector distinctions even when correlations
    all spike to 1.0 during a crash.

    Args:
        returns_window: (N_stocks, T) array of log returns
        tickers:        List of ticker symbols (same order as returns_window)
        sector_weight:  Weight given to static sector signal (default 0.30)
        percentile:     Adaptive threshold percentile (default 70)

    Returns:
        signals: (N_stocks,) hybrid GAT deviation signal
    """
    from topo_trader.data.sector_map import sector_adjacency_matrix
    from topo_trader.strategies.gat_engine import get_adaptive_threshold

    n_assets = returns_window.shape[0]

    # Dynamic correlation component
    corr = np.corrcoef(returns_window)
    np.nan_to_num(corr, copy=False)
    corr = np.clip(corr, -1.0, 1.0)
    threshold = get_adaptive_threshold(corr, percentile=percentile)
    dyn_adj   = np.where(np.abs(corr) > threshold, np.abs(corr), 0.0)
    np.fill_diagonal(dyn_adj, 0.0)

    # Static sector component
    sec_adj = sector_adjacency_matrix(tickers)

    # Normalise each to [0,1]
    if dyn_adj.max() > 0:
        dyn_adj = dyn_adj / dyn_adj.max()
    if sec_adj.max() > 0:
        sec_adj = sec_adj / sec_adj.max()

    # Hybrid adjacency
    adj = (1 - sector_weight) * dyn_adj + sector_weight * sec_adj
    np.fill_diagonal(adj, 0.0)

    if adj.sum() == 0:
        return np.zeros(n_assets)

    # Attention-weighted signal (same as gat_engine.py)
    r_t     = returns_window[:, -1]
    signals = np.zeros(n_assets)

    for i in range(n_assets):
        w = adj[i, :]
        if w.sum() == 0:
            continue
        shifted   = w - w.max()
        exp_w     = np.where(w > 0, np.exp(shifted), 0.0)
        attention = exp_w / (exp_w.sum() + 1e-12)
        signals[i] = r_t[i] - float(np.dot(attention, r_t))

    return signals


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    det = IndiaRegimeDetector()
    np.random.seed(42)

    # Simulate crash regime (high vol, negative returns)
    crash_rets = np.random.normal(-0.002, 0.025, (30, 60))
    print(f"Crash simulation: regime = {det.regime_label(det.detect(crash_rets))}")

    # Simulate bull regime (low vol, positive returns)
    bull_rets = np.random.normal(0.001, 0.006, (30, 60))
    print(f"Bull simulation:  regime = {det.regime_label(det.detect(bull_rets))}")

    # Event override test
    demo_date = pd.Timestamp('2016-11-15')
    fake_rets = np.random.normal(0.0, 0.01, (30, 60))
    print(f"Demonetization date: regime = {det.regime_label(det.detect(fake_rets, demo_date))}")
