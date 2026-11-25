# Lightweight LSTM Sequence Model for AQI Forecasting under Distribution Shift
import os
import copy
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def get_torch_device() -> torch.device:
    # Use Apple Silicon MPS acceleration if available, otherwise CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# 18 Features per time step in 14-day historical sequence
LSTM_FEATURE_COLS = [
    "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "sin_DayOfYear", "cos_DayOfYear", "sin_Month", "cos_Month",
    "sin_DayOfWeek", "cos_DayOfWeek", "is_weekend",
    "is_diwali_window", "is_stubble_burning", "is_winter_inversion", "is_lockdown"
]


class LSTMNet(nn.Module):
    """
    PyTorch 1-Layer LSTM architecture with fully connected head and dropout.
    Input: (batch_size, seq_len=14, input_dim=18)
    Output: (batch_size, 1) continuous AQI forecast
    """
    def __init__(self, input_dim: int = 18, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # (batch_size, hidden_dim)
        return self.fc(last_step)


class LSTMSequenceDatasetBuilder:
    """
    Builds sliding-window sequence tensors for full city time series.
    Guarantees:
    - Never crosses city boundaries.
    - True calendar-day historical window [t-13, t].
    - Verifies strictly consecutive daily dates without gaps.
    - Strictly causal: forward-fills past observations only; initial missing values
      are imputed strictly from training-period statistics (Date <= train_end_date),
      NEVER from future observations (zero backward filling / no bfill).
    - Zero future lookahead.
    """
    def __init__(
        self,
        feature_cols: List[str] = LSTM_FEATURE_COLS,
        seq_len: int = 14,
        handle_missing_dates: str = "reindex_ffill",
        train_end_date: str = "2018-03-13",
        imputation_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.handle_missing_dates = handle_missing_dates
        self.train_end_date = pd.to_datetime(train_end_date)
        self.imputation_stats = imputation_stats or {}

    def fit_imputation_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Fits training-period city-specific medians and global medians strictly
        on Date <= train_end_date (proper training cutoff 2018-03-13).
        Guarantees zero leakage from validation, calibration, or test periods.
        Raises ValueError if no proper training records exist on or before train_end_date.
        """
        df_tr = df.copy()
        df_tr["Date"] = pd.to_datetime(df_tr["Date"])
        tr_slice = df_tr[df_tr["Date"] <= self.train_end_date]
        if len(tr_slice) == 0:
            raise ValueError(
                f"No records found on or before train_end_date={self.train_end_date.strftime('%Y-%m-%d')} "
                "to fit imputation statistics. Sequence builder strictly forbids fitting imputation statistics "
                "on validation or test data to prevent future leakage. Please provide training-period data "
                "or initialize with explicitly supplied saved training-only imputation statistics."
            )

        stats: Dict[str, Dict[str, float]] = {"city_medians": {}, "global_medians": {}}
        for col in self.feature_cols:
            if col in tr_slice.columns:
                valid_vals = tr_slice[col].dropna()
                col_med = float(valid_vals.median()) if len(valid_vals) > 0 else 0.0
                stats["global_medians"][col] = col_med if not np.isnan(col_med) else 0.0

        for city, grp in tr_slice.groupby("City"):
            stats["city_medians"][city] = {}
            for col in self.feature_cols:
                if col in grp.columns:
                    valid_vals = grp[col].dropna()
                    cmed = float(valid_vals.median()) if len(valid_vals) > 0 else stats["global_medians"].get(col, 0.0)
                    stats["city_medians"][city][col] = cmed if not np.isnan(cmed) else stats["global_medians"].get(col, 0.0)

        self.imputation_stats = stats
        return stats

    def build_sequences_and_lookup(
        self,
        df: pd.DataFrame,
        allow_missing_dates_fallback: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[Tuple[str, pd.Timestamp], int]]:
        df_sorted = df.copy()
        df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])
        
        # Ensure training-period imputation statistics are available for causal filling
        if not self.imputation_stats or not self.imputation_stats.get("global_medians"):
            self.fit_imputation_stats(df_sorted)

        lookup: Dict[Tuple[str, pd.Timestamp], int] = {}
        all_sequences: List[np.ndarray] = []
        global_idx = 0

        mode = self.handle_missing_dates if allow_missing_dates_fallback is None else ("reindex_ffill" if allow_missing_dates_fallback else "error")

        for city, grp in df_sorted.groupby("City", sort=False):
            grp = grp.sort_values("Date").reset_index(drop=True)

            if grp["Date"].duplicated().any():
                dup_dates = grp["Date"][grp["Date"].duplicated()].dt.strftime("%Y-%m-%d").unique()
                raise ValueError(
                    f"City '{city}' has {len(dup_dates)} duplicate calendar dates: {list(dup_dates)[:3]}. "
                    "Sequence builder requires unique daily calendar observations."
                )

            diffs = grp["Date"].diff().dropna()
            is_consecutive = (diffs == pd.Timedelta(days=1)).all() if len(diffs) > 0 else True

            if not is_consecutive:
                if mode == "error":
                    irregular_gaps = grp.loc[grp["Date"].diff() > pd.Timedelta(days=1), "Date"]
                    gap_details = [
                        f"gap between {grp.at[idx-1, 'Date'].strftime('%Y-%m-%d')} and {grp.at[idx, 'Date'].strftime('%Y-%m-%d')} ({(grp.at[idx, 'Date'] - grp.at[idx-1, 'Date']).days} days)"
                        for idx in irregular_gaps.index[:3]
                    ]
                    raise ValueError(
                        f"City '{city}' contains non-consecutive calendar dates with gaps: {gap_details}. "
                        "LSTM sequence windows must represent exactly 14 consecutive calendar days. "
                        "Provide continuous calendar data or enable leakage-safe calendar reindexing."
                    )
                elif mode == "reindex_ffill":
                    # Leakage-safe reindexing to complete consecutive daily calendar
                    min_date = grp["Date"].min()
                    max_date = grp["Date"].max()
                    full_dates = pd.date_range(min_date, max_date, freq="D", name="Date")
                    
                    grp = grp.set_index("Date").reindex(full_dates)
                    grp["City"] = city
                    grp = grp.reset_index()
                else:
                    raise ValueError(f"Unknown handle_missing_dates mode: '{mode}'")

            # Strictly causal feature imputation:
            # 1. Forward-fill past observations only (NEVER backward fill from future)
            grp[self.feature_cols] = grp[self.feature_cols].ffill()

            # 2. For remaining missing values at the beginning of the series (before any observation exists),
            # impute strictly using training-period statistics (never future observations)
            city_stats = self.imputation_stats.get("city_medians", {}).get(city, {})
            global_stats = self.imputation_stats.get("global_medians", {})

            for col in self.feature_cols:
                if grp[col].isna().any():
                    fallback_val = city_stats.get(col, global_stats.get(col, 0.0))
                    grp[col] = grp[col].fillna(fallback_val)

            # 3. Explicit check: all 18 feature columns must be 100% free of missing values
            assert not grp[self.feature_cols].isna().any().any(), (
                f"City '{city}' has remaining NaNs in features after causal imputation: "
                f"{grp[self.feature_cols].isna().sum().to_dict()}"
            )

            vals = grp[self.feature_cols].to_numpy(dtype=np.float32)

            # Pad first (seq_len - 1) rows with city's first causally imputed observation
            pad = np.repeat(vals[:1], self.seq_len - 1, axis=0)
            padded = np.vstack([pad, vals])

            n_city = len(grp)
            for i in range(n_city):
                window = padded[i : i + self.seq_len]
                assert window.shape == (self.seq_len, len(self.feature_cols)), "Window shape mismatch"
                all_sequences.append(window)
                dt = grp.at[i, "Date"]
                lookup[(city, dt)] = global_idx
                global_idx += 1

        seq_tensor = np.array(all_sequences, dtype=np.float32)
        return seq_tensor, lookup


class LSTMForecasterWrapper:
    """
    Scikit-learn compatible wrapper for training, evaluating, and checkpointing LSTMNet.
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        seq_len: int = 14,
        lr: float = 0.002,
        batch_size: int = 64,
        epochs: int = 40,
        patience: int = 6,
        random_state: int = 42,
        horizon: int = 1,
    ):
        self.name = "LSTM"
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.horizon = horizon

        self.device = get_torch_device()
        self.feature_cols = LSTM_FEATURE_COLS
        self.model: Optional[LSTMNet] = None

        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

        self.builder = LSTMSequenceDatasetBuilder(self.feature_cols, self.seq_len)
        self.seq_cache: Optional[np.ndarray] = None
        self.lookup_cache: Optional[Dict[Tuple[str, pd.Timestamp], int]] = None

        self.train_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.best_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.training_duration_sec: float = 0.0

    def fit(self, full_df: pd.DataFrame, train_split: pd.DataFrame, val_split: pd.DataFrame):
        """
        Fits LSTM model using proper training data and validation-based early stopping.
        """
        t0 = time.time()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        # 1. Build or reuse continuous sequence cache
        self.seq_cache, self.lookup_cache = self.builder.build_sequences_and_lookup(full_df)

        target_col = f"target_H{self.horizon}"
        valid_col = f"target_H{self.horizon}_valid"

        # 2. Filter strictly valid targets
        tr_valid = train_split[train_split[valid_col] == 1].copy()
        val_valid = val_split[val_split[valid_col] == 1].copy()

        tr_valid["Date"] = pd.to_datetime(tr_valid["Date"])
        val_valid["Date"] = pd.to_datetime(val_valid["Date"])

        tr_indices = [self.lookup_cache[(c, d)] for c, d in zip(tr_valid["City"], tr_valid["Date"])]
        val_indices = [self.lookup_cache[(c, d)] for c, d in zip(val_valid["City"], val_valid["Date"])]

        X_tr_raw = self.seq_cache[tr_indices]
        y_tr_raw = tr_valid[target_col].to_numpy(dtype=np.float32)

        X_val_raw = self.seq_cache[val_indices]
        y_val_raw = val_valid[target_col].to_numpy(dtype=np.float32)

        # 3. Fit normalization strictly on proper training period
        # X_tr_raw shape: (N_train, seq_len, n_features)
        flat_tr_features = X_tr_raw.reshape(-1, len(self.feature_cols))
        self.mean_ = np.nanmean(flat_tr_features, axis=0)
        self.scale_ = np.nanstd(flat_tr_features, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0

        self.y_mean = float(np.nanmean(y_tr_raw))
        self.y_std = float(np.nanstd(y_tr_raw))
        if self.y_std == 0.0:
            self.y_std = 1.0

        # Apply normalization
        X_tr_norm = (X_tr_raw - self.mean_) / self.scale_
        y_tr_norm = (y_tr_raw - self.y_mean) / self.y_std

        X_val_norm = (X_val_raw - self.mean_) / self.scale_
        y_val_norm = (y_val_raw - self.y_mean) / self.y_std

        # Convert to PyTorch tensors
        X_tr_t = torch.tensor(X_tr_norm, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_norm, dtype=torch.float32).unsqueeze(1)

        X_val_t = torch.tensor(X_val_norm, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val_norm, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 4. Instantiate model and optimizer
        self.model = LSTMNet(
            input_dim=len(self.feature_cols),
            hidden_dim=self.hidden_dim,
            dropout=0.1
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.SmoothL1Loss()  # Huber loss for robustness to AQI spikes

        best_weights = None
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        patience_counter = 0

        n_samples = len(X_tr_t)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            perm = torch.randperm(n_samples)
            total_train_loss = 0.0
            total_train_samples = 0

            for i in range(0, n_samples, self.batch_size):
                b_idx = perm[i : i + self.batch_size]
                b_size = len(b_idx)
                xb = X_tr_t[b_idx].to(self.device)
                yb = y_tr_t[b_idx].to(self.device)

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * b_size
                total_train_samples += b_size

            train_loss = total_train_loss / total_train_samples
            self.train_history["train_loss"].append(train_loss)

            # Validation loss: sample-weighted across validation batches
            self.model.eval()
            total_val_loss = 0.0
            total_val_samples = 0
            n_val = len(X_val_t)
            with torch.no_grad():
                for j in range(0, n_val, self.batch_size):
                    val_b_idx = slice(j, j + self.batch_size)
                    val_xb = X_val_t[val_b_idx]
                    val_yb = y_val_t[val_b_idx]
                    val_b_size = len(val_xb)
                    val_b_loss = criterion(self.model(val_xb), val_yb)
                    total_val_loss += val_b_loss.item() * val_b_size
                    total_val_samples += val_b_size

            val_loss = total_val_loss / total_val_samples
            self.train_history["val_loss"].append(val_loss)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        self.training_duration_sec = time.time() - t0
        return self

    def predict(self, df_input: pd.DataFrame) -> np.ndarray:
        """
        Predicts continuous AQI values for given input DataFrame.
        """
        if self.model is None:
            raise RuntimeError("LSTM model has not been trained or loaded.")

        self.model.eval()
        df_in = df_input.copy()
        df_in["Date"] = pd.to_datetime(df_in["Date"])

        # Extract sequence tensor using lookup cache or dynamically build if missing
        if self.lookup_cache is None or not all((c, d) in self.lookup_cache for c, d in zip(df_in["City"], df_in["Date"])):
            seq_cache, lookup_cache = self.builder.build_sequences_and_lookup(df_in)
        else:
            seq_cache, lookup_cache = self.seq_cache, self.lookup_cache

        indices = [lookup_cache[(c, d)] for c, d in zip(df_in["City"], df_in["Date"])]
        X_raw = seq_cache[indices]
        X_norm = (X_raw - self.mean_) / self.scale_

        with torch.no_grad():
            x_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
            preds_norm = self.model(x_t).cpu().numpy().ravel()

        # Invert target normalization
        preds = (preds_norm * self.y_std) + self.y_mean
        return preds

    def save_checkpoint(self, path: str):
        """
        Saves full checkpoint dictionary with model weights, scalers, and training metadata.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "feature_cols": self.feature_cols,
            "hidden_dim": self.hidden_dim,
            "seq_len": self.seq_len,
            "mean": self.mean_,
            "scale": self.scale_,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "horizon": self.horizon,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "training_duration_sec": self.training_duration_sec,
            "train_history": self.train_history,
            "imputation_stats": getattr(self.builder, "imputation_stats", {}),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, full_df: Optional[pd.DataFrame] = None):
        """
        Loads checkpoint weights, scalers, and optionally restores sequence cache.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.feature_cols = checkpoint["feature_cols"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.seq_len = checkpoint["seq_len"]
        self.mean_ = checkpoint["mean"]
        self.scale_ = checkpoint["scale"]
        self.y_mean = checkpoint["y_mean"]
        self.y_std = checkpoint["y_std"]
        self.horizon = checkpoint["horizon"]
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.training_duration_sec = checkpoint.get("training_duration_sec", 0.0)
        self.train_history = checkpoint.get("train_history", {})
        if "imputation_stats" in checkpoint and checkpoint["imputation_stats"]:
            self.builder.imputation_stats = checkpoint["imputation_stats"]
        elif self.mean_ is not None:
            self.builder.imputation_stats = {
                "city_medians": {},
                "global_medians": {col: float(m) for col, m in zip(self.feature_cols, self.mean_)},
            }

        self.model = LSTMNet(
            input_dim=len(self.feature_cols),
            hidden_dim=self.hidden_dim,
            dropout=0.1
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        if full_df is not None:
            self.seq_cache, self.lookup_cache = self.builder.build_sequences_and_lookup(full_df)

        return self
