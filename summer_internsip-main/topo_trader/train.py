import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .models.tcn import MarketTCN

def train_model(X_train, y_train, epochs=20, batch_size=32, lr=0.001):
    """
    Train the Causal TCN.
    
    Args:
        X_train: (N, 12, 64) numpy array
        y_train: (N,) numpy array (0 or 1)
    """
    # Convert to Class Weights if imbalanced? 
    # We'll stick to standard BCELoss for now.
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    
    model = MarketTCN(num_inputs=12, num_channels=[32, 32, 32, 32], kernel_size=3, dropout=0.2)
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Tensor conversions
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device) # (N, 1)
    
    dataset_size = len(X_train)
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0
        
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_tensor[indices], y_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/dataset_size:.5f}")
        
    return model

def save_model(model, path="topo_trader/models/saved_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
