# Rigorous Unit Tests for Corrected Methodology
import pytest
import numpy as np
import pandas as pd
from src.uncertainty.conformal import SplitConformalPredictor, RollingConformalPredictor
from src.data.splitters import TemporalSplitter
from src.data.imputation import LeakageSafeImputer
from src.models.classical import get_feature_columns


def test_conformal_empty_calibration_raises_exception():
    # Test 1: Conformal calibration must reject empty calibration sets without arbitrary fallbacks
    sp = SplitConformalPredictor(alpha=0.10)
    with pytest.raises(ValueError, match="Calibration set contains no valid"):
        sp.calibrate(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))

    rcp = RollingConformalPredictor(alpha=0.10, window_size=50, horizon=1)
    dummy_df = pd.DataFrame({"Date": ["2020-01-01"], "City": ["Delhi"], "target_H1_date": ["2020-01-02"]})
    with pytest.raises(ValueError, match="Initial calibration"):
        rcp.run_rolling_evaluation(dummy_df, np.array([100.0]), np.array([100.0]), np.array([]))


def test_split_target_and_origin_boundaries_h1_and_h3():
    # Test 2: Guarantee that neither origin nor target dates cross train/val/test boundaries
    dates = pd.date_range("2015-01-01", "2020-07-01", freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "City": "Delhi",
        "AQI": np.random.uniform(50, 200, size=len(dates)),
    })

    splitter = TemporalSplitter(
        train_start="2015-01-01",
        train_end="2018-12-31",
        val_ratio=0.20,
        test_start="2019-01-01",
        test_end="2020-07-01",
    )

    for H in [1, 3]:
        df[f"target_H{H}"] = df["AQI"].shift(-H)
        df[f"target_H{H}_date"] = df["Date"].shift(-H)
        df[f"target_H{H}_valid"] = df[f"target_H{H}"].notna().astype(int)

        tr, val, ts = splitter.get_splits(df, horizon=H)

        t_date_col = f"target_H{H}_date"
        # Assert proper training target dates are strictly before validation start date
        assert (pd.to_datetime(tr[t_date_col]) < pd.to_datetime(val["Date"].min())).all()

        # Assert validation target dates are strictly before test start date
        assert (pd.to_datetime(val[t_date_col]) < pd.to_datetime(ts["Date"].min())).all()

        # Assert zero overlap in target dates
        assert len(set(tr[t_date_col]).intersection(set(val[t_date_col]))) == 0
        assert len(set(val[t_date_col]).intersection(set(ts[t_date_col]))) == 0


def test_synthetic_multi_city_rolling_conformal_with_gaps():
    # Test 3: Synthetic multi-city data with missing calendar days and missing target observations
    np.random.seed(42)
    # Staggered calendar with missing days
    dates_cityA = pd.date_range("2019-01-01", "2019-06-30", freq="2D")
    dates_cityB = pd.date_range("2019-01-03", "2019-06-30", freq="3D")
    dates_cityC = pd.date_range("2019-01-02", "2019-06-30", freq="D")

    records = []
    for c, dt_range in [("CityA", dates_cityA), ("CityB", dates_cityB), ("CityC", dates_cityC)]:
        for d in dt_range:
            records.append({"City": c, "Date": d, "AQI": np.random.uniform(30, 300)})

    synth_df = pd.DataFrame(records).sort_values(["City", "Date"]).reset_index(drop=True)

    H = 3
    synth_df[f"target_H{H}_date"] = synth_df["Date"] + pd.Timedelta(days=H)
    synth_df[f"target_H{H}"] = synth_df["AQI"] * 1.05 + np.random.normal(0, 5, size=len(synth_df))

    # Intentionally insert NaN targets to simulate missing monitoring days
    nan_mask = np.random.rand(len(synth_df)) < 0.15
    synth_df.loc[nan_mask, f"target_H{H}"] = np.nan

    predictions = synth_df[f"target_H{H}"].fillna(100.0).to_numpy() + np.random.normal(0, 10, size=len(synth_df))
    y_true = synth_df[f"target_H{H}"].to_numpy()
    cal_scores = np.abs(np.random.normal(0, 15, size=100))

    rcp = RollingConformalPredictor(alpha=0.10, window_size=50, horizon=H, mode="pooled")
    lower, upper, q_hist = rcp.run_rolling_evaluation(synth_df, predictions, y_true, cal_scores)

    assert len(lower) == len(synth_df)
    assert len(upper) == len(synth_df)
    assert (upper >= lower).all()
    assert (q_hist > 0).all()


def test_feature_columns_sanitization():
    # Test 4: Guarantee no targets, metadata, or indicators enter feature columns
    dummy_df = pd.DataFrame({
        "City": ["Delhi", "Chennai"],
        "Date": ["2018-01-01", "2018-01-02"],
        "AQI_observed": [1, 0],
        "target_H1": [120.0, 130.0],
        "target_H1_date": ["2018-01-02", "2018-01-03"],
        "target_H1_valid": [1, 1],
        "target_H3": [140.0, 150.0],
        "AQI_raw_unimputed": [110.0, 115.0],
        "AQI_Bucket": ["Moderate", "Poor"],
        "Season": ["Winter", "Winter"],
        "AQI_lag_1": [105.0, 110.0],
        "PM2.5_lag_1": [60.0, 65.0],
        "sin_Month": [0.5, 0.5],
    })

    cols = get_feature_columns(dummy_df)
    forbidden = ["AQI_observed", "target_H1", "target_H1_date", "target_H1_valid", "target_H3", "AQI_raw_unimputed", "AQI_Bucket", "Season", "City", "Date"]
    for f in forbidden:
        assert f not in cols
    assert "AQI_lag_1" in cols
    assert "PM2.5_lag_1" in cols


def test_imputation_proper_training_boundary():
    # Test 5: Verify imputation statistics are fitted strictly on proper training period
    dates = pd.date_range("2015-01-01", "2020-07-01", freq="D")
    df = pd.DataFrame({
        "City": "Delhi",
        "Date": dates,
        "AQI": np.random.uniform(50, 150, size=len(dates)),
        "PM2.5": np.random.uniform(20, 80, size=len(dates)),
    })

    # Proper train cutoff: 2018-03-13
    imputer1 = LeakageSafeImputer(method="ffill_train_median", train_end_date="2018-03-13")
    imputer1.fit(df, ["AQI", "PM2.5"])
    stat_clean = imputer1.fit_stats_["city_medians"]["Delhi"]["AQI"]

    # Perturb validation and test periods (after 2018-03-13)
    df_perturbed = df.copy()
    post_proper_train = df_perturbed["Date"] > "2018-03-13"
    df_perturbed.loc[post_proper_train, "AQI"] = 99999.0

    imputer2 = LeakageSafeImputer(method="ffill_train_median", train_end_date="2018-03-13")
    imputer2.fit(df_perturbed, ["AQI", "PM2.5"])
    stat_perturbed = imputer2.fit_stats_["city_medians"]["Delhi"]["AQI"]

    assert stat_clean == stat_perturbed, "Imputer statistics changed when post-proper-training data was modified!"
