# Unit tests for calendar alignment, leakage safety, and split partitions
import pytest
import pandas as pd
import numpy as np
from src.data.loader import load_raw_city_data, reindex_city_calendar
from src.data.splitters import TemporalSplitter
from src.data.imputation import LeakageSafeImputer
from src.features.engineer import build_full_feature_pipeline


def test_city_calendar_continuity():
    # Verify continuous daily frequency without gaps
    raw_df = load_raw_city_data()
    for city in ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Lucknow"]:
        city_df = reindex_city_calendar(raw_df, city)
        diffs = city_df["Date"].diff().dropna()
        assert (diffs == pd.Timedelta(days=1)).all(), f"City {city} has non-daily steps!"


def test_target_horizon_alignment():
    # Verify H1 is exactly t+1 day and H3 is exactly t+3 days
    raw_df = load_raw_city_data()
    imputer = LeakageSafeImputer(method="ffill_train_median", train_end_date="2018-12-31")
    imputer.fit(raw_df, ["AQI", "PM2.5"])
    feat_df = build_full_feature_pipeline(raw_df, imputer)

    for city in ["Delhi", "Bengaluru"]:
        cdf = feat_df[feat_df["City"] == city].sort_values("Date").reset_index(drop=True)
        # Check consecutive rows
        for i in range(len(cdf) - 3):
            d_curr = cdf.loc[i, "Date"]
            # Target H1 should correspond to ground truth at d_curr + 1 day
            # If target_H1_valid is 1, it must match raw unimputed AQI at d_curr + 1 day
            d_next = cdf.loc[i + 1, "Date"]
            assert d_next == d_curr + pd.Timedelta(days=1)
            if cdf.loc[i, "target_H1_valid"] == 1:
                # Compare with raw unimputed AQI at i+1
                raw_match = raw_df[(raw_df["City"] == city) & (raw_df["Date"] == d_next)]["AQI"].values[0]
                assert np.isclose(cdf.loc[i, "target_H1"], raw_match)

            d_3 = cdf.loc[i + 3, "Date"]
            assert d_3 == d_curr + pd.Timedelta(days=3)
            if cdf.loc[i, "target_H3_valid"] == 1:
                raw_match_3 = raw_df[(raw_df["City"] == city) & (raw_df["Date"] == d_3)]["AQI"].values[0]
                assert np.isclose(cdf.loc[i, "target_H3"], raw_match_3)


def test_temporal_splits_no_overlap():
    # Verify strict non-overlapping boundaries between train, val, and test for H1
    raw_df = load_raw_city_data()
    splitter = TemporalSplitter(train_start="2015-01-01", train_end="2018-12-31", test_start="2019-01-01")
    tr, val, ts = splitter.get_splits(raw_df, horizon=1)

    assert tr["Date"].max() < val["Date"].min()
    assert val["Date"].max() <= pd.to_datetime("2018-12-31")
    assert ts["Date"].min() >= pd.to_datetime("2019-01-01")
    assert len(set(tr["Date"]).intersection(set(ts["Date"]))) == 0
    assert len(set(val["Date"]).intersection(set(ts["Date"]))) == 0


def test_imputer_fitted_on_train_only():
    # Verify imputer stats do not change if validation or test data values are perturbed
    raw_df = load_raw_city_data()
    imputer1 = LeakageSafeImputer(method="ffill_train_median", train_end_date="2018-03-13")
    imputer1.fit(raw_df, ["AQI", "PM2.5"])

    # Perturb data after 2018-03-13
    perturbed_df = raw_df.copy()
    test_mask = perturbed_df["Date"] > "2018-03-13"
    perturbed_df.loc[test_mask, "AQI"] = 9999.0
    perturbed_df.loc[test_mask, "PM2.5"] = 8888.0

    imputer2 = LeakageSafeImputer(method="ffill_train_median", train_end_date="2018-03-13")
    imputer2.fit(perturbed_df, ["AQI", "PM2.5"])

    # The fitted statistics must be identical
    for city in ["Delhi", "Bengaluru"]:
        assert imputer1.fit_stats_["city_medians"][city]["AQI"] == imputer2.fit_stats_["city_medians"][city]["AQI"]
        assert imputer1.fit_stats_["city_medians"][city]["PM2.5"] == imputer2.fit_stats_["city_medians"][city]["PM2.5"]


def test_seasonal_holdout_split_purging():
    # Verify seasonal holdout: train on Mar-Oct only, test on Nov-Feb (2019-2020), purged by target date
    raw_df = load_raw_city_data()
    splitter = TemporalSplitter()

    for h in [1, 3]:
        train_df, val_df, test_df = splitter.get_seasonal_holdout_split(raw_df, horizon=h)

        # Check train target dates are in months 3-10
        tr_targets = pd.to_datetime(train_df["Date"]) + pd.Timedelta(days=h)
        assert tr_targets.dt.month.isin([3, 4, 5, 6, 7, 8, 9, 10]).all(), f"H{h}: Train target dates outside Mar-Oct!"

        # Check val target dates are in months 3-10
        val_targets = pd.to_datetime(val_df["Date"]) + pd.Timedelta(days=h)
        assert val_targets.dt.month.isin([3, 4, 5, 6, 7, 8, 9, 10]).all(), f"H{h}: Val target dates outside Mar-Oct!"

        # Check test target dates are in Nov-Feb (months 11, 12, 1, 2) in 2019-2020
        ts_targets = pd.to_datetime(test_df["Date"]) + pd.Timedelta(days=h)
        assert ts_targets.dt.month.isin([11, 12, 1, 2]).all(), f"H{h}: Test target dates outside Nov-Feb!"
        assert ts_targets.dt.year.isin([2019, 2020]).all(), f"H{h}: Test target years outside 2019-2020!"

        # Target date purge check: no train or val target date enters test origin period
        ts_origins = pd.to_datetime(test_df["Date"])
        assert (tr_targets < ts_origins.min()).all(), f"H{h}: Train target dates overlap test origins!"
        assert (val_targets < ts_origins.min()).all(), f"H{h}: Val target dates overlap test origins!"

