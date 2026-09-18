"""
TopoTrader V3 — Portfolio Allocator (Week 9)
=============================================
Combines signals across all 49 Nifty stocks into a real portfolio.

Algorithm (each trading day):
  1. Get V3 model probabilities for all 49 stocks
  2. Apply RiskManager deadband per regime
  3. Rank by confidence (distance from 0.5)
  4. Take top-K stocks (K = 5, 7, or 10 configurable)
  5. Allocate capital using Kelly weights
  6. Apply stop-losses from StopLossManager

Key metrics computed:
  - Sharpe Ratio (annualised, per regime)
  - Max Drawdown
  - CAGR
  - Win Rate
  - Avg holding period

Usage:
    from topo_trader.strategies.portfolio_allocator import PortfolioAllocator
    alloc = PortfolioAllocator(top_k=7)
    weights = alloc.allocate(signals)   # list[TradeSignal]
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from topo_trader.strategies.risk_manager import TradeSignal, RiskManager

# ── Constants ─────────────────────────────────────────────────────────────────
ROUNDTRIP_COST  = 0.0036    # 0.36% round-trip NSE
ANNUAL_DAYS     = 252
RISK_FREE_RATE  = 0.00      # 0% — benchmarked against market-neutral (signal system)
DAILY_RISK_FREE = 0.0       # raw return/volatility ratio (Sharpe vs cash = 0)

# Empirical Nifty-50 daily return magnitudes (2010-2021)
MEAN_UP_RETURN  =  0.0067
MEAN_DN_RETURN  = -0.0070


@dataclass
class DayPortfolio:
    """State of portfolio on one trading day."""
    date:         pd.Timestamp
    positions:    Dict[str, float]   # ticker -> weight
    cash:         float = 1.0        # uninvested cash fraction
    regime:       int   = -1


class PortfolioAllocator:
    """
    Top-K portfolio allocator using Kelly-weighted signals.

    Args:
        top_k:          Max number of simultaneous positions (default 7)
        kelly_fraction: Fractional Kelly multiplier (default 0.5)
        allow_short:    Whether to allow short positions (default True)
        max_pos_size:   Max single position as fraction of portfolio (default 0.20)
    """

    def __init__(self, top_k: int = 7, kelly_fraction: float = 1.0,
                 allow_short: bool = True, max_pos_size: float = 0.20):
        self.top_k         = top_k
        self.allow_short   = allow_short
        self.max_pos_size  = max_pos_size
        self.rm            = RiskManager(kelly_fraction=kelly_fraction)

    def allocate(self, signals: List[TradeSignal],
                 rolling_hr: float = 0.512,
                 avg_win: float = MEAN_UP_RETURN,
                 avg_loss: float = abs(MEAN_DN_RETURN)) -> Dict[str, float]:
        """
        Convert list of trade signals into portfolio weights.

        Returns:
            weights: dict ticker -> signed weight (+long, -short)
                     Weights sum to at most 1.0 in absolute value.
        """
        # Filter out HOLDs
        active = [s for s in signals if s.direction != 'HOLD']

        if not active:
            return {}

        if not self.allow_short:
            active = [s for s in active if s.direction == 'LONG']

        # Sort by confidence (highest first)
        active.sort(key=lambda s: s.confidence, reverse=True)

        # Take top-K
        selected = active[:self.top_k]

        # Compute Kelly sizes
        weights = {}
        total_alloc = 0.0

        for sig in selected:
            sig.kelly_frac = self.rm.kelly.compute(rolling_hr, avg_win, avg_loss)
            # Scale by confidence
            scale = min(1.0 + sig.confidence * 2, 1.5)
            size  = min(sig.kelly_frac * scale, self.max_pos_size)

            if total_alloc + size > 1.0:
                size = max(1.0 - total_alloc, 0.0)

            if size < 0.01:
                break

            signed = size if sig.direction == 'LONG' else -size
            weights[sig.ticker] = signed
            total_alloc += size

        return weights


class PortfolioBacktester:
    """
    Walk-forward portfolio backtester.

    Given per-ticker V3 model probabilities across time, simulates
    daily portfolio construction and tracks P&L.

    Returns equity curve, Sharpe, MaxDD, CAGR, Win Rate.
    """

    def __init__(self, allocator: Optional[PortfolioAllocator] = None):
        self.allocator = allocator or PortfolioAllocator(top_k=7)

    def run(self, prob_df: pd.DataFrame, return_df: pd.DataFrame,
            regime_series: pd.Series,
            initial_capital: float = 1_000_000.0) -> Dict:
        """
        Run full portfolio backtest.

        Args:
            prob_df:       DataFrame (dates x tickers) of model probabilities
            return_df:     DataFrame (dates x tickers) of actual next-day returns
            regime_series: Series (dates) of regime labels (0-3)
            initial_capital: Starting portfolio value

        Returns:
            dict with equity_curve, sharpe, max_dd, cagr, win_rate, per_regime_sharpe
        """
        capital    = initial_capital
        equity_arr = [capital]
        dates      = [prob_df.index[0]]
        daily_rets = []
        trades     = []
        regime_rets= {0: [], 1: [], 2: [], 3: []}

        self.allocator.rm.sl.reset()
        prev_weights: Dict[str, float] = {}   # track existing positions
        day_counter = 0                        # for weekly rebalancing
        REBAL_FREQ  = 5                        # rebalance every 5 trading days

        for date in prob_df.index[:-1]:
            if date not in return_df.index:
                continue

            regime = int(regime_series.get(date, -1))
            probs  = prob_df.loc[date].dropna()
            rets   = return_df.loc[date].dropna()
            day_counter += 1

            common = probs.index.intersection(rets.index)
            if len(common) == 0:
                equity_arr.append(equity_arr[-1])
                dates.append(date)
                daily_rets.append(0.0)
                continue

            # ── Weekly rebalancing: only rebuild portfolio every 5 days ──────
            # On non-rebalance days, hold prev_weights unchanged (zero cost)
            if day_counter % REBAL_FREQ == 1 or not prev_weights:
                # Rebalance day — build fresh signals
                signals = []
                for ticker in common:
                    prob = float(probs[ticker])
                    sig  = self.allocator.rm.evaluate(prob, regime)
                    sig.ticker = ticker
                    signals.append(sig)
                new_weights = self.allocator.allocate(signals)
            else:
                # Hold day — keep existing positions, no rebalancing cost
                new_weights = prev_weights

            # P&L — return on existing positions; costs only on position CHANGES
            day_pnl = 0.0
            all_tickers = set(new_weights) | set(prev_weights)

            for ticker in all_tickers:
                old_w = prev_weights.get(ticker, 0.0)
                new_w = new_weights.get(ticker, 0.0)
                if ticker not in rets.index:
                    continue
                raw_ret = float(rets[ticker])

                # Return on whichever weight is being held today
                hold_w = new_w                          # hold the new weight
                day_pnl += hold_w * raw_ret

                # Cost only on the delta (opening/closing/changing)
                delta = abs(new_w - old_w)
                day_pnl -= delta * ROUNDTRIP_COST       # one-way cost on each side

                if abs(new_w) > 1e-4:
                    correct = (new_w > 0 and raw_ret > 0) or \
                              (new_w < 0 and raw_ret < 0)
                    trades.append({
                        'date': date, 'ticker': ticker,
                        'direction': 'LONG' if new_w > 0 else 'SHORT',
                        'weight': abs(new_w), 'return': new_w * raw_ret,
                        'correct': correct,
                    })

            prev_weights = new_weights

            capital *= (1 + day_pnl)
            equity_arr.append(capital)
            dates.append(date)
            daily_rets.append(day_pnl)

            if regime in regime_rets:
                regime_rets[regime].append(day_pnl)

            # Check circuit breakers
            status = self.allocator.rm.sl.update(day_pnl)
            if status['suspended']:
                # Zero returns while suspended
                for _ in range(min(21, len(prob_df) - len(equity_arr))):
                    equity_arr.append(equity_arr[-1])
                    daily_rets.append(0.0)
                prev_weights = {}   # reset positions during suspension

        # ── Metrics ──────────────────────────────────────────────────────────
        eq_arr   = np.array(equity_arr)
        ret_arr  = np.array(daily_rets)

        sharpe   = self._sharpe(ret_arr)
        max_dd   = self._max_drawdown(eq_arr)
        n_years  = max(len(ret_arr) / ANNUAL_DAYS, 0.01)
        cagr     = (eq_arr[-1] / eq_arr[0]) ** (1 / n_years) - 1

        trade_df = pd.DataFrame(trades)
        win_rate = trade_df['correct'].mean() if len(trade_df) > 0 else 0.5

        # Per-regime Sharpe
        per_regime_sharpe = {}
        regime_names = {0: 'Crash', 1: 'High-Vol', 2: 'Bull', 3: 'Sideways'}
        for r, rrets in regime_rets.items():
            per_regime_sharpe[regime_names[r]] = self._sharpe(np.array(rrets)) \
                                                  if len(rrets) > 5 else 0.0

        return {
            'equity_curve':       pd.Series(eq_arr, index=dates[:len(eq_arr)]),
            'daily_returns':      ret_arr,
            'sharpe':             sharpe,
            'max_drawdown':       max_dd,
            'cagr':               cagr,
            'win_rate':           win_rate,
            'n_trades':           len(trades),
            'final_capital':      float(eq_arr[-1]),
            'per_regime_sharpe':  per_regime_sharpe,
            'trade_log':          trade_df,
        }

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        if len(returns) < 5 or returns.std() < 1e-8:
            return 0.0
        excess = returns - DAILY_RISK_FREE
        return float(excess.mean() / excess.std() * np.sqrt(ANNUAL_DAYS))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if len(equity) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity)
        dd   = (equity - peak) / (peak + 1e-10)
        return float(dd.min())

    def print_summary(self, results: Dict):
        print("\n" + "=" * 60)
        print("PORTFOLIO BACKTEST SUMMARY")
        print("=" * 60)
        print(f"  Sharpe Ratio (annualised)  : {results['sharpe']:+.3f}")
        print(f"  Max Drawdown               : {results['max_drawdown']:.2%}")
        print(f"  CAGR                       : {results['cagr']:.2%}")
        print(f"  Win Rate                   : {results['win_rate']:.2%}")
        print(f"  Total trades               : {results['n_trades']:,}")
        print(f"  Final capital              : {results['final_capital']:,.0f}")
        print("\n  Per-Regime Sharpe:")
        for regime, sharpe in results['per_regime_sharpe'].items():
            print(f"    {regime:<12}: {sharpe:+.3f}")
