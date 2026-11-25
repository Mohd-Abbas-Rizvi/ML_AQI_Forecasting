# Comprehensive Validation and Reproducibility Test Suite
import pytest
import numpy as np
import pandas as pd
import joblib
import os

from src.data.loader import load_and_prepare_all_cities, TARGET_CITIES
from src.data.splitters import TemporalSplitter
from src.data.imputation import LeakageSafeImputer
from src.features.engineer import build_full_feature_pipeline
from src.models.classical import ModelFactory, get_feature_columns
from src.models.scratch_numpy import NumPyRidgeRegression, NumPyMLPRegressor
from src.uncertainty.conformal import SplitConformalPredictor, RollingConformalPredictor
from src.evaluation.metrics import calculate_point_metrics, calculate_interval_metrics


def test_chronological_ordering_and_horizons():
    # Verify strict calendar daily continuity and target alignments
    raw_df = load_and_prepare_all_cities()
    imputer = LeakageSafeImputer(method="ffill_train_median", train_end_date="2018-12-31")
    imputer.fit(raw_df, ["AQI", "PM2.5"])
    df = build_full_feature_pipeline(raw_df, imputer)

    for city in TARGET_CITIES:
        cdf = df[df["City"] == city].sort_values("Date").reset_index(drop=True)
        # Verify strictly monotonically increasing dates by 1 day
        date_diffs = cdf["Date"].diff().dropna()
        assert (date_diffs == pd.Timedelta(days=1)).all()

        # Verify H1 and H3 horizons
        for i in range(len(cdf) - 3):
            d_t = cdf.loc[i, "Date"]
            d_next = cdf.loc[i + 1, "Date"]
            d_3 = cdf.loc[i + 3, "Date"]
            assert d_next == d_t + pd.Timedelta(days=1)
            assert d_3 == d_t + pd.Timedelta(days=3)


def test_no_future_leakage_in_features():
    # Verify features at t do not contain future AQI or target values
    df = pd.read_csv("data/processed/features_ffill_train_median.csv")
    cols = get_feature_columns(df)

    # Future targets must never be in predictive feature columns
    for forbidden in ["target_H1", "target_H3", "AQI_raw_unimputed"]:
        assert forbidden not in cols


def test_conformal_calibration_guarantee():
    # Verify split conformal prediction achieves nominal coverage on exchangeable holdout
    np.random.seed(42)
    y_true = np.random.normal(100, 20, size=1000)
    y_pred = y_true + np.random.normal(0, 10, size=1000)

    # 500 cal, 500 test
    y_cal, p_cal = y_true[:500], y_pred[:500]
    y_ts, p_ts = y_true[500:], y_pred[500:]

    sp = SplitConformalPredictor(alpha=0.10)
    sp.calibrate(y_cal, p_cal)
    lo, up = sp.predict_intervals(p_ts)
    metrics = calculate_interval_metrics(y_ts, lo, up, nominal=0.90)

    # Coverage on exchangeable sample must be within binomial sampling error (~88% to 92%)
    assert 0.87 <= metrics["PICP"] <= 0.93


def test_model_prediction_shapes():
    # Verify all model outputs match input sample length
    df = pd.read_csv("data/processed/features_ffill_train_median.csv")
    cols = get_feature_columns(df)
    X = df.iloc[:100][cols].values

    for model_name in ["Persistence", "Ridge", "GradientBoosting", "NumPy_Ridge"]:
        mpath = f"models/{model_name}_H1.joblib"
        if os.path.exists(mpath):
            m = joblib.load(mpath)
            if model_name == "Persistence":
                preds = m.predict(df.iloc[:100])
            else:
                preds = m.predict(X)
            assert len(preds) == 100
            assert isinstance(preds, np.ndarray)


def test_reproducible_model_outputs():
    # Verify deterministic reproducibility with fixed random seed
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    m1 = NumPyRidgeRegression(alpha=10.0)
    m1.fit(X, y)
    p1 = m1.predict(X)

    m2 = NumPyRidgeRegression(alpha=10.0)
    m2.fit(X, y)
    p2 = m2.predict(X)

    assert np.allclose(p1, p2)
