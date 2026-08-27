import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .models.tcn import MarketTCN


def train_model(X_train, y_train, epochs=30, batch_size=64, lr=1e-3, num_inputs=None):
    """
    Train the Causal TCN with cosine LR annealing and early stopping.

    Args:
        X_train   : (N, num_inputs, window_len) numpy array — channels already normalized
        y_train   : (N,) numpy array (0 or 1)
        epochs    : max epochs (early stopping may cut short)
        batch_size: mini-batch size
        lr        : initial learning rate (decays to lr/100 via cosine schedule)
        num_inputs: number of input channels; inferred from X_train if None
    """
    if num_inputs is None:
        num_inputs = X_train.shape[1]   # infer from data — no hardcoding

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...", flush=True)

    model = MarketTCN(num_inputs=num_inputs,
                      num_channels=[32, 32, 32, 32],
                      kernel_size=3, dropout=0.2)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Cosine annealing: lr decays smoothly from lr → lr/100 over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    dataset_size = len(X_train)

    # Early stopping state
    best_loss = float('inf')
    patience  = 5       # epochs without improvement before stopping
    min_delta = 1e-5    # minimum meaningful improvement
    wait      = 0

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        epoch_loss  = 0.0
        batches_run = 0

        for i in range(0, dataset_size, batch_size):
            indices          = permutation[i : i + batch_size]
            batch_x, batch_y = X_tensor[indices], y_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss    = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping — prevents exploding gradients with 16-channel input
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss  += loss.item()
            batches_run += 1

        scheduler.step()

        # Correct display: loss per batch (true BCE mean), not divided by dataset size
        mean_loss = epoch_loss / max(batches_run, 1)
        print(f"Epoch {epoch+1:>3}/{epochs} - Loss: {mean_loss:.5f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        # Early stopping
        if mean_loss < best_loss - min_delta:
            best_loss = mean_loss
            wait      = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [EarlyStopping] No improvement for {patience} epochs. "
                      f"Stopping at epoch {epoch+1}.", flush=True)
                break

    return model


def save_model(model, path="topo_trader/models/saved_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
