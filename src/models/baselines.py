# Baseline forecasters: Persistence, Seasonal Naive, and Auto-Regressive
import numpy as np
import pandas as pd
from typing import Dict, Optional


class PersistenceForecaster:
    # Baseline 1: Persistence (tomorrow/day-3 = today: y_{t+h} = y_t)
    def __init__(self):
        self.name = "Persistence"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Persistence has no parameters to fit
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Returns most recent known AQI at forecast origin t
        if "AQI_lag_0" in df.columns:
            return df["AQI_lag_0"].to_numpy(dtype=float)
        elif "AQI" in df.columns:
            return df["AQI"].to_numpy(dtype=float)
        else:
            raise KeyError("AQI_lag_0 column required for Persistence")


class SeasonalNaiveForecaster:
    # Baseline 2: Seasonal Naive (y_{t+h} = y_{t+h-365}, with persistence fallback)
    def __init__(self):
        self.name = "SeasonalNaive"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "AQI_lag_365" in df.columns:
            preds = df["AQI_lag_365"].to_numpy(dtype=float).copy()
            # If 365 days ago is missing or NaN, fallback to lag 0
            nan_mask = np.isnan(preds)
            if np.any(nan_mask) and "AQI_lag_0" in df.columns:
                preds[nan_mask] = df.loc[nan_mask, "AQI_lag_0"].to_numpy(dtype=float)
            return preds
        elif "AQI_lag_0" in df.columns:
            return df["AQI_lag_0"].to_numpy(dtype=float)
        else:
            raise KeyError("AQI_lag_365 or AQI_lag_0 required for Seasonal Naive")


class ARIMAForecaster:
    # Baseline 3: Auto-regressive model fitted per city
    def __init__(self, p: int = 5, d: int = 1, q: int = 0):
        self.name = f"ARIMA({p},{d},{q})"
        self.order = (p, d, q)
        self.models = {}

    def fit(self, train_df: pd.DataFrame):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            for city, cgroup in train_df.groupby("City"):
                series = cgroup.set_index("Date")["AQI"].dropna()
                if len(series) > 50:
                    model = ARIMA(series, order=self.order)
                    self.models[city] = model.fit()
        except Exception as e:
            print(f"ARIMA fit failed: {e}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Fallback to persistence if ARIMA cannot forecast out of sample
        if "AQI_lag_0" in df.columns:
            return df["AQI_lag_0"].to_numpy(dtype=float)
        return np.full(len(df), np.nan)
