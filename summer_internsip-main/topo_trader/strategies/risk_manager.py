"""
TopoTrader V3 — Risk Manager (Week 8)
======================================
Converts binary buy/sell signals into proper position sizes
with downside protection.

Components:
  1. KellyCriterion       — optimal position size from rolling hit rate
  2. DynamicDeadband      — per-regime confidence thresholds (asymmetric)
  3. StopLossManager      — per-trade and portfolio circuit breakers

Usage:
    from topo_trader.strategies.risk_manager import RiskManager
    rm = RiskManager()
    size = rm.position_size(prob=0.63, regime=2, rolling_hr=0.512, avg_win=0.0067, avg_loss=0.0070)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── NSE Transaction costs (Zerodha) ──────────────────────────────────────────
ROUNDTRIP_COST = 0.0036   # 0.36% round-trip (brokerage + STT + GST + stamp)

# ── Per-regime deadband thresholds ────────────────────────────────────────────
# Asymmetric: short bar is lower because stocks fall faster/further than they rise.
# Regime indices match C13-C16 one-hot: 0=Crash, 1=HighVol, 2=Bull, 3=Sideways
REGIME_DEADBANDS = {
    0: (0.62, 0.38),   # Crash:    only very confident calls
    1: (0.58, 0.42),   # High-Vol: moderately confident
    2: (0.54, 0.46),   # Bull:     lower bar — momentum is easier to predict
    3: (0.56, 0.44),   # Sideways: medium bar
    -1: (0.55, 0.45),  # Unknown / default
}

# ── Stop-loss thresholds ──────────────────────────────────────────────────────
PER_TRADE_STOP  = 0.015   # exit if position moves -1.5% against you
WEEKLY_STOP     = 0.030   # exit all if portfolio -3% in a week
MONTHLY_CIRCUIT = 0.050   # pause trading if portfolio -5% in a month
MAX_KELLY_FRAC  = 0.20    # never bet more than 20% of portfolio on one stock


@dataclass
class TradeSignal:
    ticker:     str
    direction:  str            # 'LONG', 'SHORT', 'HOLD'
    prob:       float          # model raw probability
    confidence: float          # distance from 0.5
    regime:     int            # 0=Crash, 1=HV, 2=Bull, 3=Sideways
    kelly_frac: float = 0.0    # recommended position size (fraction of portfolio)
    stop_loss:  float = 0.0    # stop-loss level (absolute move)


class KellyCriterion:
    """
    Kelly Criterion position sizer.

    f* = (p*b - q) / b
      p = model hit rate (rolling 30-day window)
      q = 1 - p
      b = avg_win / avg_loss (reward-to-risk ratio)

    Uses fractional Kelly (50%) for conservative sizing to account for
    model uncertainty and non-stationarity.
    """

    def __init__(self, fraction: float = 0.5, floor: float = 0.01,
                 cap: float = MAX_KELLY_FRAC):
        self.fraction = fraction   # fractional Kelly multiplier
        self.floor    = floor      # minimum position if signal is active
        self.cap      = cap        # maximum position

    def compute(self, hit_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Args:
            hit_rate:  Rolling hit rate of the model (0.50-0.55 range typically)
            avg_win:   Average winning trade return (e.g. 0.0067 = +0.67%)
            avg_loss:  Average losing trade return magnitude (e.g. 0.0070 = -0.70%)

        Returns:
            kelly_frac: Recommended position as fraction of portfolio (0–0.20)
        """
        if avg_loss < 1e-6:
            return self.floor

        b = avg_win / avg_loss           # reward-to-risk ratio
        q = 1.0 - hit_rate
        raw_kelly = (hit_rate * b - q) / b

        # Apply fractional Kelly and clamp
        sized = self.fraction * raw_kelly
        return float(np.clip(sized, self.floor, self.cap))


class DynamicDeadband:
    """
    Regime-aware, asymmetric confidence deadband.

    Long threshold > 0.5, Short threshold < 0.5.
    Short bar is lower because:
      - Stocks fall faster when sentiment breaks
      - Short signals need less certainty to be profitable

    Returns 'LONG', 'SHORT', or 'HOLD'.
    """

    def __init__(self, custom_bands: Optional[dict] = None):
        self.bands = custom_bands or REGIME_DEADBANDS

    def classify(self, prob: float, regime: int) -> str:
        long_thr, short_thr = self.bands.get(regime, self.bands[-1])
        if prob >= long_thr:
            return 'LONG'
        elif prob <= short_thr:
            return 'SHORT'
        return 'HOLD'

    def confidence(self, prob: float, direction: str) -> float:
        """Distance from decision boundary (0.5)."""
        return abs(prob - 0.5)


class StopLossManager:
    """
    Tracks portfolio P&L and triggers stop-losses.

    Per-trade:  Immediately exit if return crosses -per_trade_stop
    Weekly:     Exit all positions if weekly portfolio return < -weekly_stop
    Monthly:    Suspend trading if monthly return < -monthly_circuit
    """

    def __init__(self, per_trade: float = PER_TRADE_STOP,
                 weekly: float = WEEKLY_STOP,
                 monthly: float = MONTHLY_CIRCUIT):
        self.per_trade = per_trade
        self.weekly    = weekly
        self.monthly   = monthly

        self._week_start_val  = 1.0
        self._month_start_val = 1.0
        self._portfolio_val   = 1.0
        self._day             = 0
        self._suspended       = False

    def reset(self):
        self._week_start_val  = 1.0
        self._month_start_val = 1.0
        self._portfolio_val   = 1.0
        self._day             = 0
        self._suspended       = False

    def update(self, daily_return: float) -> dict:
        """
        Update portfolio value and check circuit breakers.

        Args:
            daily_return: Portfolio return for the day (e.g. +0.005 = +0.5%)

        Returns:
            dict with keys: 'active' (bool), 'exit_all' (bool), 'suspended' (bool)
        """
        self._portfolio_val  *= (1 + daily_return)
        self._day            += 1

        # Weekly reset (every 5 trading days)
        if self._day % 5 == 0:
            self._week_start_val = self._portfolio_val

        # Monthly reset (every 21 trading days)
        if self._day % 21 == 0:
            self._month_start_val = self._portfolio_val
            self._suspended = False  # re-enable after a month

        weekly_ret  = self._portfolio_val / self._week_start_val  - 1
        monthly_ret = self._portfolio_val / self._month_start_val - 1

        exit_all  = weekly_ret  < -self.weekly
        suspended = monthly_ret < -self.monthly

        if suspended:
            self._suspended = True

        return {
            'active':    not self._suspended,
            'exit_all':  exit_all,
            'suspended': self._suspended,
            'portfolio_val': self._portfolio_val,
            'weekly_ret':    weekly_ret,
            'monthly_ret':   monthly_ret,
        }

    def should_stop_trade(self, trade_return: float) -> bool:
        """True if this individual trade hit its stop-loss."""
        return trade_return < -self.per_trade


class RiskManager:
    """
    Combined risk management system.

    Wraps KellyCriterion + DynamicDeadband + StopLossManager into one interface.

    Typical usage:
        rm = RiskManager()
        rm.sl.reset()

        for each day:
            for each ticker:
                signal = rm.evaluate(prob, regime, rolling_hr, avg_win, avg_loss)
                if signal.direction != 'HOLD':
                    execute trade with signal.kelly_frac size
            status = rm.sl.update(portfolio_daily_return)
            if status['exit_all'] or status['suspended']:
                close all positions
    """

    def __init__(self, kelly_fraction: float = 0.5):
        self.kelly    = KellyCriterion(fraction=kelly_fraction)
        self.deadband = DynamicDeadband()
        self.sl       = StopLossManager()

    def evaluate(self, prob: float, regime: int,
                 rolling_hr: float = 0.512,
                 avg_win:    float = 0.0067,
                 avg_loss:   float = 0.0070) -> TradeSignal:
        """
        Evaluate one stock's model output into a full trade signal.

        Args:
            prob:       Model output probability (0-1)
            regime:     Current market regime (0=Crash,1=HV,2=Bull,3=Sideways)
            rolling_hr: Model's rolling 30-day hit rate (default 51.2%)
            avg_win:    Mean return on winning trades
            avg_loss:   Mean absolute return on losing trades

        Returns:
            TradeSignal with direction, kelly fraction, and stop-loss level
        """
        direction  = self.deadband.classify(prob, regime)
        confidence = self.deadband.confidence(prob, direction)

        if direction == 'HOLD':
            kelly_frac = 0.0
        else:
            kelly_frac = self.kelly.compute(rolling_hr, avg_win, avg_loss)
            # Scale kelly by confidence (more confident = larger position)
            scale      = min(1.0 + confidence * 2, 1.5)   # max 1.5x kelly
            kelly_frac = min(kelly_frac * scale, MAX_KELLY_FRAC)

        return TradeSignal(
            ticker    = '',
            direction = direction,
            prob      = prob,
            confidence= confidence,
            regime    = regime,
            kelly_frac= kelly_frac,
            stop_loss = PER_TRADE_STOP,
        )


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rm = RiskManager(kelly_fraction=0.5)

    print("RiskManager self-test")
    print("=" * 55)
    test_cases = [
        (0.65, 0, "High confidence, Crash regime"),
        (0.58, 2, "Moderate confidence, Bull regime"),
        (0.52, 2, "Low confidence, Bull regime (expect HOLD)"),
        (0.38, 1, "Short signal, High-Vol regime"),
        (0.50, 3, "Neutral, Sideways (expect HOLD)"),
    ]
    for prob, regime, desc in test_cases:
        sig = rm.evaluate(prob, regime, rolling_hr=0.512)
        print(f"  {desc}")
        print(f"    prob={prob:.2f}  regime={regime}  -> "
              f"{sig.direction}  kelly={sig.kelly_frac:.3f}  stop={sig.stop_loss:.3f}")

    # Stop-loss manager test
    print("\nStop-loss manager test (simulating -4% weekly drawdown):")
    rm.sl.reset()
    for d in range(7):
        ret = -0.006 if d < 6 else 0.001
        status = rm.sl.update(ret)
        print(f"  Day {d+1}: weekly={status['weekly_ret']:.2%}  "
              f"exit_all={status['exit_all']}  suspended={status['suspended']}")
