import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ─── DATA PREPARATION FOR SEQUENCE MODELS ────────────────────────────────────

def create_sequences(data: np.ndarray, lookback: int = 72, horizon: int = 24):
    """
    Convert a time series into (X_seq, y) pairs for LSTM input.
    
    lookback=72  → use last 72 hours (3 days) of features as input
    horizon=24   → predict AQI 24 hours ahead
    
    Returns:
        X: shape (n_samples, lookback, n_features)
        y: shape (n_samples,)
    """
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i, :])      # past window
        y.append(data[i + horizon - 1, 0])        # AQI at t+horizon (col 0)
    return np.array(X), np.array(y)


class AQIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── LSTM MODEL ───────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    Stacked LSTM with dropout for AQI time series forecasting.
    
    Architecture:
      Input (batch, lookback, n_features)
        └─ LSTM Layer 1 (hidden_dim=128, return sequences)
        └─ Dropout (0.2)
        └─ LSTM Layer 2 (hidden_dim=64, return last hidden state)
        └─ Dropout (0.2)
        └─ Fully Connected (64 → 32 → 1)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)       # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]          # take last timestep
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ─── TRANSFORMER MODEL ────────────────────────────────────────────────────────

class TransformerForecaster(nn.Module):
    """
    Lightweight Transformer encoder for AQI forecasting.
    
    Why Transformer for AQI?
    - Attention mechanism captures long-range dependencies (e.g., weekly patterns)
    - Better than LSTM at learning which past hours matter most
    - Interpretable via attention weights
    """
    def __init__(self, input_dim: int, d_model: int = 64, 
                 nhead: int = 4, num_layers: int = 2, 
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)               # (batch, seq_len, d_model)
        x = self.transformer_encoder(x)       # (batch, seq_len, d_model)
        x = x.mean(dim=1)                     # global average pooling
        return self.fc_out(x).squeeze(-1)


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def train_deep_model(model, X_train, y_train, X_val, y_val,
                     epochs=100, batch_size=64, lr=1e-3, patience=15):
    """
    PyTorch training loop with:
    - Adam optimizer + ReduceLROnPlateau scheduler
    - Early stopping (stops when val loss stops improving)
    - Best model checkpointing
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)     # robust to AQI outliers vs MSELoss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5, verbose=True
    )

    train_ds = AQIDataset(X_train, y_train)
    val_ds   = AQIDataset(X_val,   y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)  # no shuffle for TS!
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_losses = []
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                val_losses.append(criterion(pred, y_b).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    return model, history
