# Classical Machine Learning models: Ridge, Random Forest, Gradient Boosting, XGBoost
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    # Select all valid numerical predictive feature columns, strictly excluding targets, identifiers, and metadata
    forbidden_prefixes = ("target_", "target", "raw_", "label_")
    forbidden_exact = {
        "City", "Date", "AQI_raw_unimputed", "AQI_raw", "AQI_observed",
        "AQI_Bucket", "Season", "pm2.5_raw", "target_H1", "target_H3"
    }

    cols = []
    for c in df.columns:
        if c in forbidden_exact or any(c.startswith(p) for p in forbidden_prefixes):
            continue
        if df[c].dtype in [np.float64, np.int64, float, int]:
            cols.append(c)

    # Sort deterministically
    return sorted(cols)


class ModelFactory:
    # Builds standard models with fixed random seeds for reproducibility
    @staticmethod
    def create_model(model_name: str, random_state: int = 42):
        if model_name == "Ridge":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=10.0, random_state=random_state))
            ])
        elif model_name == "RandomForest":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_split=5,
                    random_state=random_state,
                    n_jobs=1
                ))
            ])
        elif model_name == "GradientBoosting":
            return HistGradientBoostingRegressor(
                max_iter=120,
                max_depth=6,
                learning_rate=0.08,
                random_state=random_state
            )
        elif model_name == "XGBoost":
            return xgb.XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=1
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
