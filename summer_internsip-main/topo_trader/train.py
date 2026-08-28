"""
TopoTrader V3 — Training Pipeline
===================================
Supports:
    - V3 model (Dual-Branch TCN + SE-Net + MoE)
    - Magnitude-weighted BCE loss
    - Incremental fine-tuning (freeze backbone, retrain heads only)
    - Same cosine LR + gradient clipping + early stopping as V2

Usage
-----
# Full training (Week 6):
model = train_model_v3(X_train, y_train, magnitudes_train)

# Fine-tuning from prior window (Week 11):
model = train_model_v3(X_train, y_train, magnitudes_train,
                       pretrained_model=prev_model, finetune=True)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .models.tcn import MarketTCN
from .models.tcn_v3 import MarketTCN_V3, MagnitudeWeightedBCE


# ─────────────────────────────────────────────────────────────────────────────
# V2 trainer (unchanged — kept for fair comparison runs)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, epochs=30, batch_size=64, lr=1e-3,
                num_inputs=None):
    """
    V2 training function — flat MarketTCN, plain BCE loss.
    Kept intact for ablation / baseline comparison.
    """
    if num_inputs is None:
        num_inputs = X_train.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[V2] Training on {device}...", flush=True)

    model = MarketTCN(num_inputs=num_inputs,
                      num_channels=[32, 32, 32, 32],
                      kernel_size=3, dropout=0.2)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                      eta_min=lr / 100)
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    best_loss, patience, min_delta, wait = float('inf'), 5, 1e-5, 0
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train))
        epoch_loss, batches = 0.0, 0
        for i in range(0, len(X_train), batch_size):
            idx     = perm[i: i + batch_size]
            bx, by  = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            loss    = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item(); batches += 1
        scheduler.step()
        mean_loss = epoch_loss / max(batches, 1)
        print(f"  Epoch {epoch+1:>3}/{epochs}  Loss: {mean_loss:.5f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)
        if mean_loss < best_loss - min_delta:
            best_loss = mean_loss; wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [EarlyStopping] Stopped at epoch {epoch+1}.", flush=True)
                break
    return model


# ─────────────────────────────────────────────────────────────────────────────
# V3 trainer — new
# ─────────────────────────────────────────────────────────────────────────────

def train_model_v3(
    X_train,
    y_train,
    magnitudes_train = None,   # (N,) abs returns — enables magnitude-weighted loss
    epochs           = 30,
    batch_size       = 64,
    lr               = 1e-3,
    pretrained_model = None,   # pass a MarketTCN_V3 to fine-tune
    finetune         = False,  # if True, freeze TCN backbone, only train MoE heads
):
    """
    Train MarketTCN_V3 with:
        - Dual-Branch TCN + SE-Net + MoE (in tcn_v3.py)
        - Magnitude-weighted BCE loss
        - Cosine LR annealing (1e-3 → 1e-5)
        - Gradient clipping (max_norm=1.0)
        - Early stopping (patience=5)
        - Optional fine-tuning: freeze backbone, retrain MoE heads only

    Parameters
    ----------
    X_train          : (N, 16, T) feature tensor — must have 16 channels
    y_train          : (N,) binary labels (0/1)
    magnitudes_train : (N,) absolute next-day returns for weighting (optional)
    pretrained_model : existing MarketTCN_V3 to start from (fine-tune mode)
    finetune         : freeze branch_a, branch_b, se — only train moe

    Returns
    -------
    model : trained MarketTCN_V3 on CPU
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode   = "FINETUNE" if finetune else "FULL TRAIN"
    print(f"[V3 {mode}] Device: {device}", flush=True)

    # ── Build or reuse model ───────────────────────────────────────────────
    if pretrained_model is not None:
        model = pretrained_model
        print(f"  Starting from pretrained weights.", flush=True)
    else:
        model = MarketTCN_V3(
            n_tech_ch   = 7,
            n_topo_ch   = 9,
            branch_ch   = [32, 32, 32, 32],
            kernel_size = 3,
            dropout     = 0.2,
            se_reduction= 4,
            n_experts   = 4,
            n_regime_ch = 4,
        )

    model.to(device)

    # ── Fine-tuning: freeze backbone ───────────────────────────────────────
    if finetune:
        for name, param in model.named_parameters():
            if name.startswith('moe') or name.startswith('se'):
                param.requires_grad = True    # train attention + experts
            else:
                param.requires_grad = False   # freeze TCN branches
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Fine-tune mode: {trainable:,} trainable params "
              f"(MoE + SE-Net only).", flush=True)
        # Lower LR for fine-tuning
        lr = lr / 10

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion  = MagnitudeWeightedBCE()

    # ── Optimiser (only optimise requires_grad=True params) ───────────────
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    # ── Move data to device ───────────────────────────────────────────────
    X_t   = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t   = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    if magnitudes_train is not None:
        mag_t = torch.tensor(magnitudes_train, dtype=torch.float32).to(device)
    else:
        mag_t = None

    N = len(X_train)

    # ── Training loop ─────────────────────────────────────────────────────
    best_loss, patience, min_delta, wait = float('inf'), 10, 1e-5, 0
    model.train()

    for epoch in range(epochs):
        perm        = torch.randperm(N)
        epoch_loss  = 0.0
        batches     = 0

        for i in range(0, N, batch_size):
            idx      = perm[i: i + batch_size]
            bx, by   = X_t[idx], y_t[idx]
            bmag     = mag_t[idx] if mag_t is not None else None

            optimizer.zero_grad()
            preds    = model(bx)
            loss     = criterion(preds, by, bmag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches    += 1

        scheduler.step()
        mean_loss = epoch_loss / max(batches, 1)
        print(f"  Epoch {epoch+1:>3}/{epochs}  Loss: {mean_loss:.5f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        # Early stopping
        if mean_loss < best_loss - min_delta:
            best_loss = mean_loss
            wait      = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [EarlyStopping] Stopped at epoch {epoch+1}.", flush=True)
                break

    model.cpu()  # move back to CPU for evaluation / saving
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"  Model saved → {path}", flush=True)


def load_model_v3(path, device='cpu'):
    """Load a saved V3 model for inference or fine-tuning."""
    model = MarketTCN_V3()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
