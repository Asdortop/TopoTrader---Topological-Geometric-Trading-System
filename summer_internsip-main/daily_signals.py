"""
TopoTrader V3 — Daily Signal Generator (Week 11)
=================================================
Full live/paper-trading signal pipeline.

Steps:
  1. Load the most recent trained V3 model (latest walk-forward window)
  2. Download or load latest NSE data
  3. Compute 16-channel features for all available tickers
  4. Run V3 inference -> probabilities for all stocks
  5. Apply RiskManager (regime-aware deadband + Kelly sizing)
  6. Output: console table + HTML dashboard

Usage:
    python daily_signals.py
    python daily_signals.py --date 2021-04-15    # specific date (backtesting)
    python daily_signals.py --top-k 5            # change number of signals

Output:
    Console: formatted signal table
    reports/signal_dashboard.html  — shareable HTML report
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topo_trader.models.tcn_v3 import MarketTCN_V3
from topo_trader.strategies.risk_manager import RiskManager, TradeSignal
from topo_trader.strategies.regime_detector_india import IndiaRegimeDetector
from topo_trader.utils.data_loader import (
    fetch_india_csv_tickers, load_india_csv_data, generate_features,
)

os.makedirs("reports", exist_ok=True)

WINDOW_LEN       = 64
DEFAULT_TOP_K    = 7
DEFAULT_MODEL    = "topo_trader/models/tcn_v3_w5_COVID_Crash.pth"  # latest window

REGIME_NAMES     = {0: 'Crash', 1: 'High-Vol', 2: 'Bull', 3: 'Sideways'}
DIR_EMOJI        = {'LONG': '+', 'SHORT': '-', 'HOLD': ' '}
DIR_COLOR_HTML   = {'LONG': '#2ecc71', 'SHORT': '#e74c3c', 'HOLD': '#95a5a6'}


# ─────────────────────────────────────────────────────────────────────────────
# Feature loading
# ─────────────────────────────────────────────────────────────────────────────

def load_features_for_date(target_date: pd.Timestamp):
    """
    Load feature data, returning only data available up to target_date.
    Simulates "what you would have known on that date" (no lookahead).
    """
    tickers = fetch_india_csv_tickers()
    data = load_india_csv_data(start_date="2010-01-01",
                               end_date=target_date.strftime("%Y-%m-%d"),
                               cache_name="india_csv_data.parquet")
    available_tickers = list(data.columns.get_level_values(0).unique())
    features, _ = generate_features(data, available_tickers, parallel=False, n_jobs=1)
    return features, available_tickers, data


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> MarketTCN_V3:
    model = MarketTCN_V3()
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state)
        print(f"Loaded model: {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}. Using untrained model.")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model: MarketTCN_V3, features: dict, tickers: list,
                  target_date: pd.Timestamp) -> dict:
    """
    Run model inference for all tickers on target_date.

    Returns:
        dict: ticker -> probability (0-1)
    """
    probs = {}
    model.eval()

    for t in tickers:
        if t not in features:
            continue
        df = features[t]
        mask = df.index <= target_date
        dr = df.loc[mask]
        if len(dr) < WINDOW_LEN:
            continue   # not enough history

        window = dr.values[-WINDOW_LEN:].T   # (16, 64)
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 16, 64)
        with torch.no_grad():
            p = float(model(x).flatten()[0])
        probs[t] = p

    return probs


# ─────────────────────────────────────────────────────────────────────────────
# Regime detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_current_regime(features: dict, tickers: list,
                           target_date: pd.Timestamp) -> int:
    """Detect market regime from feature data around target_date."""
    detector = IndiaRegimeDetector()
    returns_list = []

    for t in tickers[:20]:  # Use first 20 tickers for efficiency
        if t not in features:
            continue
        df = features[t]
        mask = (df.index <= target_date)
        dr = df.loc[mask]
        if len(dr) < 20 or 'C1_LogRet' not in dr.columns:
            continue
        returns_list.append(dr['C1_LogRet'].values[-60:])

    if not returns_list:
        return 3  # default sideways

    min_len = min(len(r) for r in returns_list)
    if min_len < 5:
        return 3

    arr = np.array([r[-min_len:] for r in returns_list])   # (N, T)
    return detector.detect(arr, date=target_date)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling hit rate (from saved results)
# ─────────────────────────────────────────────────────────────────────────────

def load_rolling_hit_rate() -> float:
    """Load the most recent pooled hit rate from saved results."""
    paths = [
        "reports/walk_forward_v3_india.csv",
        "reports/pooled_significance_v3.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if 'v3_hr' in df.columns:
                    return float(df['v3_hr'].mean())
            except Exception:
                pass
    return 0.512   # default from pooled results


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(probs: dict, regime: int, top_k: int,
                     rolling_hr: float = 0.512) -> list:
    """
    Convert probabilities to ranked trade signals.

    Returns:
        List of TradeSignal objects, sorted by confidence (highest first)
    """
    rm = RiskManager(kelly_fraction=0.5)
    signals = []

    for ticker, prob in probs.items():
        sig = rm.evaluate(prob, regime, rolling_hr=rolling_hr)
        sig.ticker = ticker
        if sig.direction != 'HOLD':
            signals.append(sig)

    # Sort by confidence
    signals.sort(key=lambda s: s.confidence, reverse=True)

    # Return top-K
    return signals[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_console_table(signals: list, regime: int, target_date: pd.Timestamp,
                         rolling_hr: float):
    """Print formatted signal table to console."""
    regime_name = REGIME_NAMES.get(regime, 'Unknown')

    print("\n" + "=" * 75)
    print(f"  TopoTrader V3 — Daily Signals")
    print(f"  Date   : {target_date.strftime('%d %b %Y')}")
    print(f"  Regime : {regime_name}  |  Model HR: {rolling_hr:.2%}")
    print("=" * 75)
    print(f"  {'#':<3} {'Ticker':<14} {'Dir':<6} {'Prob':>6} {'Conf':>6} "
          f"{'Kelly%':>7} {'Stop':>6}")
    print("-" * 75)

    if not signals:
        print("  No confident signals today (market may be in uncertain state).")
    else:
        for i, sig in enumerate(signals, 1):
            d = DIR_EMOJI[sig.direction]
            print(f"  {i:<3} {sig.ticker:<14} {d}{sig.direction:<5} "
                  f"{sig.prob:>6.3f} {sig.confidence:>6.3f} "
                  f"{sig.kelly_frac*100:>6.1f}% {sig.stop_loss*100:>5.1f}%")

    print("=" * 75)
    print(f"  Active signals: {len(signals)}  |  "
          f"Long: {sum(1 for s in signals if s.direction=='LONG')}  |  "
          f"Short: {sum(1 for s in signals if s.direction=='SHORT')}")
    print("=" * 75)


def generate_html_dashboard(signals: list, regime: int,
                             target_date: pd.Timestamp, rolling_hr: float,
                             all_probs: dict, output_path: str = "reports/signal_dashboard.html"):
    """Generate an HTML signal dashboard."""
    regime_name = REGIME_NAMES.get(regime, 'Unknown')
    regime_color = {'Crash': '#e74c3c', 'High-Vol': '#e67e22',
                    'Bull': '#2ecc71', 'Sideways': '#3498db'}.get(regime_name, '#95a5a6')
    date_str = target_date.strftime('%d %B %Y')

    # Signal rows
    if signals:
        signal_rows = ""
        for i, sig in enumerate(signals, 1):
            color = DIR_COLOR_HTML[sig.direction]
            bg    = '#1a2744' if i % 2 == 0 else '#1e2d52'
            signal_rows += f"""
            <tr style="background:{bg}">
                <td style="color:#aaa">{i}</td>
                <td><strong>{sig.ticker}</strong></td>
                <td style="color:{color};font-weight:bold">{sig.direction}</td>
                <td>{sig.prob:.3f}</td>
                <td>{sig.confidence:.3f}</td>
                <td style="color:#f39c12">{sig.kelly_frac*100:.1f}%</td>
                <td style="color:#e74c3c">-{sig.stop_loss*100:.1f}%</td>
            </tr>"""
    else:
        signal_rows = '<tr><td colspan="7" style="text-align:center;color:#888">No confident signals today</td></tr>'

    # All stocks probability bars
    prob_bars = ""
    sorted_probs = sorted(all_probs.items(), key=lambda x: abs(x[1]-0.5), reverse=True)[:20]
    for ticker, prob in sorted_probs:
        pct   = int(prob * 100)
        bar_w = int(abs(prob - 0.5) * 200)
        bar_color = '#2ecc71' if prob > 0.5 else '#e74c3c'
        signal_in = any(s.ticker == ticker for s in signals)
        highlight = 'border-left:3px solid #f39c12;' if signal_in else ''
        prob_bars += f"""
        <div style="display:flex;align-items:center;margin:4px 0;{highlight}padding-left:4px">
            <span style="width:100px;color:#ccc;font-size:13px">{ticker}</span>
            <div style="width:200px;background:#1a2744;border-radius:3px;overflow:hidden">
                <div style="width:{bar_w}px;height:14px;background:{bar_color};
                             margin-left:{'100px' if prob > 0.5 else str(100-bar_w)+'px'}"></div>
            </div>
            <span style="margin-left:8px;color:#aaa;font-size:12px">{prob:.3f}</span>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TopoTrader V3 — Signal Dashboard — {date_str}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1526; color:#e0e6f0; min-height:100vh; }}
  .container {{ max-width:900px; margin:0 auto; padding:24px 16px; }}
  h1 {{ font-size:24px; font-weight:700; color:#fff; }}
  h2 {{ font-size:15px; font-weight:600; color:#8899bb; margin:20px 0 10px; text-transform:uppercase; letter-spacing:1px; }}
  .header {{ background:linear-gradient(135deg,#1a2744,#0d1526); border:1px solid #2a3a5e; border-radius:12px; padding:20px 24px; margin-bottom:20px; }}
  .meta {{ display:flex; gap:24px; margin-top:12px; flex-wrap:wrap; }}
  .meta-item {{ background:#0d1526; border-radius:8px; padding:10px 16px; }}
  .meta-label {{ font-size:11px; color:#8899bb; text-transform:uppercase; letter-spacing:0.5px; }}
  .meta-value {{ font-size:18px; font-weight:700; color:#fff; margin-top:2px; }}
  .regime-badge {{ display:inline-block; background:{regime_color}; color:#fff; border-radius:20px; padding:4px 12px; font-size:13px; font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th {{ background:#0d1526; padding:10px 12px; text-align:left; color:#8899bb; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; }}
  td {{ padding:10px 12px; border-bottom:1px solid #1a2744; }}
  .card {{ background:#131f3a; border:1px solid #1e2d52; border-radius:10px; padding:20px; margin-bottom:16px; }}
  .footer {{ text-align:center; color:#4a5a7a; font-size:12px; margin-top:20px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>TopoTrader V3 — Signal Dashboard</h1>
    <div class="meta">
      <div class="meta-item"><div class="meta-label">Date</div><div class="meta-value">{date_str}</div></div>
      <div class="meta-item"><div class="meta-label">Regime</div><div class="meta-value"><span class="regime-badge">{regime_name}</span></div></div>
      <div class="meta-item"><div class="meta-label">Model Hit Rate</div><div class="meta-value">{rolling_hr:.2%}</div></div>
      <div class="meta-item"><div class="meta-label">Active Signals</div><div class="meta-value">{len(signals)}</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Top Signals</h2>
    <table>
      <thead><tr>
        <th>#</th><th>Ticker</th><th>Direction</th><th>Probability</th>
        <th>Confidence</th><th>Position Size</th><th>Stop Loss</th>
      </tr></thead>
      <tbody>{signal_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>All Stocks — Probability Heatmap (Top 20 by conviction)</h2>
    <div style="display:flex;align-items:center;margin-bottom:8px;font-size:11px;color:#8899bb">
      <span style="width:100px"></span>
      <span style="width:100px;text-align:right">SHORT</span>
      <span style="width:20px;text-align:center">|</span>
      <span style="width:100px;text-align:left">LONG</span>
    </div>
    {prob_bars}
  </div>

  <div class="card">
    <h2>Risk Parameters (Active)</h2>
    <table>
      <thead><tr><th>Parameter</th><th>Value</th><th>Note</th></tr></thead>
      <tbody>
        <tr><td>Per-trade stop-loss</td><td>-1.5%</td><td>Immediate exit</td></tr>
        <tr><td>Weekly portfolio stop</td><td>-3.0%</td><td>Exit all positions</td></tr>
        <tr><td>Monthly circuit breaker</td><td>-5.0%</td><td>Suspend trading 21 days</td></tr>
        <tr><td>Max single position</td><td>20%</td><td>Kelly Criterion cap</td></tr>
        <tr><td>Transaction cost (round-trip)</td><td>0.36%</td><td>NSE / Zerodha</td></tr>
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generated by TopoTrader V3 &mdash; Dual-Branch TCN + SE-Net + Mixture of Experts<br>
    Statistical significance: Hit Rate 51.23%, p = 0.0001 vs NSE break-even
  </div>
</div>
</body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Dashboard saved -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TopoTrader V3 Daily Signal Generator")
    parser.add_argument('--date',    type=str, default=None,
                        help="Target date YYYY-MM-DD (default: most recent in data)")
    parser.add_argument('--model',   type=str, default=DEFAULT_MODEL,
                        help="Path to trained V3 model .pth file")
    parser.add_argument('--top-k',   type=int, default=DEFAULT_TOP_K,
                        help="Number of top signals to output (default: 7)")
    parser.add_argument('--no-html', action='store_true',
                        help="Skip HTML dashboard generation")
    args = parser.parse_args()

    # Parse target date
    if args.date:
        target_date = pd.Timestamp(args.date)
    else:
        target_date = pd.Timestamp('2021-04-15')  # last date in dataset

    print(f"\nTopoTrader V3 — Signal Generator")
    print(f"Target date : {target_date.strftime('%d %b %Y')}")
    print(f"Top-K       : {args.top_k}")
    print(f"Model       : {args.model}\n")

    # Step 1: Load features
    print("Step 1/5: Loading features ...", flush=True)
    features, tickers, _ = load_features_for_date(target_date)
    print(f"  {len(features)} tickers available with history up to {target_date.date()}")

    # Step 2: Load model
    print("\nStep 2/5: Loading V3 model ...", flush=True)
    model = load_model(args.model)

    # Step 3: Detect regime
    print("\nStep 3/5: Detecting market regime ...", flush=True)
    regime = detect_current_regime(features, tickers, target_date)
    print(f"  Regime: {REGIME_NAMES.get(regime, 'Unknown')} ({regime})")

    # Step 4: Run inference
    print("\nStep 4/5: Running inference on all tickers ...", flush=True)
    all_probs = run_inference(model, features, tickers, target_date)
    print(f"  Got probabilities for {len(all_probs)} tickers")

    # Step 5: Generate signals
    print("\nStep 5/5: Generating signals ...", flush=True)
    rolling_hr = load_rolling_hit_rate()
    signals    = generate_signals(all_probs, regime, top_k=args.top_k,
                                  rolling_hr=rolling_hr)

    # Output
    print_console_table(signals, regime, target_date, rolling_hr)

    if not args.no_html:
        generate_html_dashboard(signals, regime, target_date, rolling_hr,
                                all_probs)


if __name__ == "__main__":
    main()
