# PyTorch Neural models: Multi-Layer Perceptron and LSTM Sequence Model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PyTorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class PyTorchMLPWrapper:
    def __init__(self, hidden_dim: int = 64, lr: float = 0.005, epochs: int = 30, batch_size: int = 128, random_state: int = 42):
        self.name = "PyTorch_MLP"
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = torch.device("cpu") # Fast vectorized CPU on Apple Silicon
        self.model = None
        self.mean_ = None
        self.scale_ = None
        self.col_medians_ = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32).copy()
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        valid_y = ~np.isnan(y).ravel()
        X, y = X[valid_y], y[valid_y]

        self.col_medians_ = np.nanmedian(X, axis=0)
        self.col_medians_[np.isnan(self.col_medians_)] = 0.0
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.col_medians_, inds[1])

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0

        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) if np.std(y) > 0 else 1.0

        X_norm = (X - self.mean_) / self.scale_
        y_norm = (y - self.y_mean) / self.y_std

        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32)
        n_samples = len(X_tensor)

        self.model = PyTorchMLP(input_dim=X.shape[1], hidden_dim=self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            indices = torch.randperm(n_samples)
            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                xb, yb = X_tensor[batch_idx], y_tensor[batch_idx]
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = np.asarray(X, dtype=np.float32).copy()
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.col_medians_, inds[1])
        X_norm = (X - self.mean_) / self.scale_
        with torch.no_grad():
            xb = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
            preds_norm = self.model(xb).cpu().numpy().ravel()
        return (preds_norm * self.y_std) + self.y_mean
