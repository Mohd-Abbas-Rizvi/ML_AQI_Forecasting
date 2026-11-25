# Quantile Regression forecaster using Gradient Boosting
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.ensemble import HistGradientBoostingRegressor


class QuantileRegressorWrapper:
    # Fits lower (alpha=0.05), median (alpha=0.50), and upper (alpha=0.95) regressors
    def __init__(self, random_state: int = 42):
        self.name = "Quantile_GBM"
        self.random_state = random_state
        self.model_lower = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.05, max_iter=80, max_depth=6, random_state=random_state
        )
        self.model_median = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.50, max_iter=80, max_depth=6, random_state=random_state
        )
        self.model_upper = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.95, max_iter=80, max_depth=6, random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[valid], y[valid]

        self.model_lower.fit(X, y)
        self.model_median.fit(X, y)
        self.model_upper.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        lower = np.maximum(0.0, self.model_lower.predict(X))
        median = np.maximum(0.0, self.model_median.predict(X))
        upper = np.maximum(0.0, self.model_upper.predict(X))
        # Ensure monotonic bounds: lower <= median <= upper
        lower = np.minimum(lower, median)
        upper = np.maximum(upper, median)
        return lower, median, upper
