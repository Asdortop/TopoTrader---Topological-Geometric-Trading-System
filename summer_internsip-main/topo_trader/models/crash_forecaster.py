"""
TopoTrader V2 -- Topological Crash Probability Forecaster
=========================================================
A standalone LSTM model that watches the 30-day trajectory of persistence
entropy features and outputs forward-looking crash probabilities at three horizons.

Architecture:
  Input  : (batch, 5, 30) -- last 30 days of [H0, H1, ΔH0, market_ret, |market_ret|]
  LSTM   : 2-layer, hidden_size=32
  Output : (batch, 3) -- P(crash_5d), P(crash_10d), P(crash_20d)

Key upgrade over V1:
  V1 binary veto: if H0_today < 1.0 -> go to cash   (reactive, single threshold)
  V2 forecaster : watches entropy trajectory         (predictive, probabilistic)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os


# -- Model ----------------------------------------------------------------------

class CrashForecaster(nn.Module):
    """
    LSTM-based topological crash probability forecaster.

    Input  : (batch, features=5, seq_len=30)  <- note: features-first, not time-first
    Output : (batch, 3)  -> [P_5day, P_10day, P_20day]
    """

    def __init__(self, input_features: int = 5, hidden_size: int = 32,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Three separate prediction heads -- one per horizon
        def _head():
            return nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        self.head_5d  = _head()
        self.head_10d = _head()
        self.head_20d = _head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, features, seq_len) -- transpose to (batch, seq_len, features) for LSTM
        Returns:
            probs: (batch, 3) -- [P_5d, P_10d, P_20d]
        """
        x = x.permute(0, 2, 1)          # -> (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]          # Last time step hidden state (batch, hidden)

        p5  = self.head_5d(h)           # (batch, 1)
        p10 = self.head_10d(h)
        p20 = self.head_20d(h)

        return torch.cat([p5, p10, p20], dim=1)  # (batch, 3)


# -- Feature / Label Construction -----------------------------------------------

def prepare_crash_features(h0_vec: np.ndarray, h1_vec: np.ndarray,
                            market_returns: np.ndarray, window: int = 30) -> tuple:
    """
    Build feature sequences for the crash forecaster.

    Features per time step (5 total):
        [H0_entropy, H1_entropy, ΔH0 (rate of change), market_return, |market_return|]

    Args:
        h0_vec: (n_days,) H0 persistence entropy time series.
        h1_vec: (n_days,) H1 persistence entropy time series.
        market_returns: (n_days,) market (SPY / NIFTY) log returns.
        window: lookback window in days (default 30).

    Returns:
        X: (n_samples, 5, window) feature array.
        valid_indices: (n_samples,) original time indices.
    """
    n        = len(h0_vec)
    delta_h0 = np.gradient(h0_vec)
    vol      = np.abs(market_returns)

    X, idx = [], []
    for t in range(window, n):
        feat = np.stack([
            h0_vec[t-window:t],
            h1_vec[t-window:t],
            delta_h0[t-window:t],
            market_returns[t-window:t],
            vol[t-window:t],
        ], axis=0)                      # (5, window)
        X.append(feat)
        idx.append(t)

    return np.array(X, dtype=np.float32), np.array(idx)


def prepare_crash_labels(market_returns: np.ndarray,
                          crash_threshold: float = -0.05,
                          horizons: list = [5, 10, 20]) -> np.ndarray:
    """
    Create binary crash labels: did cumulative market return fall below threshold
    in the next N days?

    Args:
        market_returns: (n_days,) daily log returns.
        crash_threshold: cumulative return below this = crash event (default −5%).
        horizons: list of forecast horizons in days.

    Returns:
        labels: (n_days, len(horizons)) binary array.
    """
    n      = len(market_returns)
    labels = np.zeros((n, len(horizons)), dtype=np.float32)

    for i, horizon in enumerate(horizons):
        for t in range(n - horizon):
            cum_ret = float(np.sum(market_returns[t+1 : t+1+horizon]))
            labels[t, i] = 1.0 if cum_ret < crash_threshold else 0.0

    return labels


# -- Training -------------------------------------------------------------------

def train_crash_forecaster(h0_vec: np.ndarray, h1_vec: np.ndarray,
                            market_returns: np.ndarray,
                            train_end_idx: int,
                            window: int = 30,
                            epochs: int = 40,
                            lr: float = 5e-4,
                            save_path: str = None) -> "CrashForecaster":
    """
    Train the crash forecaster on historical data up to train_end_idx.

    Args:
        h0_vec, h1_vec: TDA entropy time series (n_days,).
        market_returns: Market daily returns (n_days,).
        train_end_idx: Last index to include in training (exclusive of test).
        window: Feature sequence length (default 30).
        epochs: Training epochs.
        lr: Learning rate.
        save_path: Optional .pth path to save trained model.

    Returns:
        Trained CrashForecaster model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CrashForecaster] Training on {device}...")

    X_all, valid_idx = prepare_crash_features(h0_vec, h1_vec, market_returns, window)
    y_all            = prepare_crash_labels(market_returns)

    # Align labels with the valid indices (offset by `window`)
    y_aligned = y_all[valid_idx, :]

    train_mask   = valid_idx <= train_end_idx
    X_train      = X_all[train_mask]
    y_train      = y_aligned[train_mask]

    if len(X_train) == 0:
        raise ValueError("No training samples -- check train_end_idx vs window size.")

    print(f"[CrashForecaster] Training samples: {len(X_train)}")
    print(f"[CrashForecaster] Crash rate (5d): {y_train[:, 0].mean():.2%}")

    model = CrashForecaster(input_features=5, hidden_size=32,
                             num_layers=2, dropout=0.3)
    model.to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    # Weighted BCE to handle class imbalance (crashes are rare)
    pos_weight = torch.tensor([(1 - y_train[:, i].mean()) / (y_train[:, i].mean() + 1e-6)
                                for i in range(3)], dtype=torch.float32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)              # (N, 3)
        loss  = 0.0
        for i in range(3):
            bce  = nn.BCELoss()(preds[:, i], y_t[:, i])
            loss = loss + bce * pos_weight[i]
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} - Loss: {loss.item():.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[CrashForecaster] Saved to {save_path}")

    return model


# -- Inference ------------------------------------------------------------------

def predict_crash_probability(model: "CrashForecaster",
                               h0_vec: np.ndarray,
                               h1_vec: np.ndarray,
                               market_returns: np.ndarray,
                               window: int = 30) -> np.ndarray:
    """
    Predict crash probabilities for the most recent window.

    Returns:
        probs: np.ndarray([P_5day, P_10day, P_20day])
    """
    model.eval()
    n        = len(h0_vec)
    delta_h0 = np.gradient(h0_vec)
    vol      = np.abs(market_returns)

    feat = np.stack([
        h0_vec[-window:],
        h1_vec[-window:],
        delta_h0[-window:],
        market_returns[-window:],
        vol[-window:],
    ], axis=0)                          # (5, window)

    device = next(model.parameters()).device
    x = torch.tensor(feat[np.newaxis], dtype=torch.float32).to(device)  # (1, 5, window)

    with torch.no_grad():
        probs = model(x).squeeze().cpu().numpy()

    return probs                        # [P_5d, P_10d, P_20d]


def load_crash_forecaster(path: str) -> "CrashForecaster":
    """Load a saved CrashForecaster model from disk."""
    model = CrashForecaster(input_features=5, hidden_size=32,
                             num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
