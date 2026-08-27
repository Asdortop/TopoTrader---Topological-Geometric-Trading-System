"""
TopoTrader V2 -- Realistic Backtester
=====================================
Upgrade over V1 backtester (backtester.py):
  - Slippage: 0.05% per trade (execution vs close price)
  - Commission: $0.50 flat per trade
  - Short borrow cost: 1% annualised (applied daily)
  - Earnings filter: skip trades within 2 days of earnings announcement
  - Crash probability gate: uses V2 forecaster when available (falls back to V1 veto)
"""

import numpy as np
import torch

# -- Transaction Cost Constants -------------------------------------------------
SLIPPAGE_PCT      = 0.0005          # 0.05% of notional per trade
COMMISSION_FLAT   = 0.50            # $0.50 flat per trade (one-way)
BORROW_ANNUAL     = 0.01            # 1% annual borrow cost for short positions
BORROW_DAILY      = BORROW_ANNUAL / 252  # Per-day borrow cost

# -- Risk Gate Constants --------------------------------------------------------
CRASH_ENTROPY_THRESHOLD   = 1.0     # V1 topological veto (fallback)
CRASH_PROB_THRESHOLD      = 0.65    # V2 forecaster: halt if P(crash_5d) > 65%
LONG_DEADBAND             = 0.55    # TCN prob > 0.55 -> long
SHORT_DEADBAND            = 0.45    # TCN prob < 0.45 -> short
TARGET_VOL                = 0.20    # 20% annualised target volatility
MAX_LEVERAGE              = 2.0     # Cap position at 2× capital
EARNINGS_BUFFER_DAYS      = 2       # Skip ±2 days around earnings


def compute_transaction_cost(position_size: float, is_short: bool = False) -> float:
    """
    Compute one-way transaction cost for a new position.

    Args:
        position_size: Absolute dollar value of the trade.
        is_short: Whether this is a short position (adds borrow cost).

    Returns:
        Total cost in dollars (always positive).
    """
    notional = abs(position_size)
    slippage  = notional * SLIPPAGE_PCT
    borrow    = notional * BORROW_DAILY if is_short else 0.0
    return slippage + COMMISSION_FLAT + borrow


def backtest_v2(
    model,
    data_tensor,
    current_volatility: float,
    capital: float = 100_000,
    h0_entropy: float = None,
    days_to_earnings: int = None,
    prev_position: float = 0.0,
    crash_prob_5d: float = None,
) -> tuple[float, float]:
    """
    V2 Risk Management and Execution Pipeline.

    Gate sequence:
      1. Crash Probability Forecaster (V2) -- overrides everything
      2. Topological Veto (V1 fallback)
      3. Earnings Filter
      4. TCN prediction -> Deadband filter
      5. Inverse Volatility Sizing
      6. Transaction cost deduction (on new/flipped trades only)

    Args:
        model: Trained MarketTCN (16-channel input).
        data_tensor: (1, 16, 64) input tensor for today.
        current_volatility: Normalised ATR for this asset.
        capital: Dollar capital allocated to this asset.
        h0_entropy: Today's H0 persistence entropy (market-wide).
        days_to_earnings: Days until (positive) or since (negative) earnings.
                          None = no filter applied.
        prev_position: Previous signed position size (used to detect new trades).
        crash_prob_5d: Output of CrashForecaster for 5-day horizon. None = skip gate.

    Returns:
        (net_position_size, transaction_cost)
          net_position_size: Signed dollars (+long, −short, 0=flat)
          transaction_cost: Dollar cost paid this step.
    """
    model.eval()

    # -- GATE 1: Crash Probability Forecaster ---------------------------------
    if crash_prob_5d is not None and crash_prob_5d > CRASH_PROB_THRESHOLD:
        return 0.0, 0.0  # High crash probability -- go to cash

    # -- GATE 2: Topological Veto (V1 fallback) -------------------------------
    if h0_entropy is not None and h0_entropy < CRASH_ENTROPY_THRESHOLD:
        return 0.0, 0.0  # H0 entropy collapsed -- market in crash regime

    # -- GATE 3: Earnings Filter -----------------------------------------------
    if days_to_earnings is not None and abs(days_to_earnings) <= EARNINGS_BUFFER_DAYS:
        return 0.0, 0.0  # Too close to earnings -- skip to avoid announcement risk

    # -- GATE 4: TCN Prediction ------------------------------------------------
    with torch.no_grad():
        if not isinstance(data_tensor, torch.Tensor):
            data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
        if data_tensor.dim() == 2:
            data_tensor = data_tensor.unsqueeze(0)  # -> (1, C, L)
        prob = model(data_tensor).item()

    # -- GATE 5: Deadband Filter -----------------------------------------------
    if prob > LONG_DEADBAND:
        signal = 1
    elif prob < SHORT_DEADBAND:
        signal = -1
    else:
        return 0.0, 0.0  # Uncertain -- stay neutral

    # -- GATE 6: Inverse Volatility Sizing ------------------------------------
    if current_volatility <= 0:
        return 0.0, 0.0
    vol_factor   = min(TARGET_VOL / current_volatility, MAX_LEVERAGE)
    gross_size   = signal * vol_factor * capital

    # -- GATE 7: Transaction Cost ----------------------------------------------
    position_changed = (prev_position == 0.0) or (np.sign(gross_size) != np.sign(prev_position))
    if position_changed:
        cost     = compute_transaction_cost(gross_size, is_short=(signal == -1))
        net_size = gross_size - np.sign(gross_size) * cost
    else:
        cost     = 0.0
        net_size = gross_size

    return float(net_size), float(cost)


def compute_pnl(position: float, next_return: float, is_short: bool = False) -> float:
    """
    Compute realised PnL for one step given a position and the actual next-day return.

    Args:
        position: Signed dollar position (+long, −short).
        next_return: Actual log return of the asset next day.
        is_short: If True, apply daily borrow cost.

    Returns:
        Realised PnL in dollars.
    """
    raw_pnl   = position * next_return
    borrow    = abs(position) * BORROW_DAILY if is_short else 0.0
    return raw_pnl - borrow
