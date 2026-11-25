# Time-series and distribution shift splitting protocols
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class TemporalSplitter:
    # Handles strict chronological train, validation, and test splits
    # Guarantees that neither forecast origin dates nor forecast target dates cross partition boundaries
    def __init__(
        self,
        train_start: str = "2015-01-01",
        train_end: str = "2018-12-31",
        val_ratio: float = 0.20,
        test_start: str = "2019-01-01",
        test_end: str = "2020-07-01",
    ):
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.val_ratio = val_ratio
        self.test_start = pd.to_datetime(test_start)
        self.test_end = pd.to_datetime(test_end)

    def get_val_cutoff_date(self, df: pd.DataFrame) -> pd.Timestamp:
        dates = pd.to_datetime(df["Date"])
        full_train_dates = np.sort(df[(dates >= self.train_start) & (dates <= self.train_end)]["Date"].unique())
        val_cutoff_idx = int(len(full_train_dates) * (1.0 - self.val_ratio))
        return pd.to_datetime(full_train_dates[val_cutoff_idx])

    def get_splits(
        self, df: pd.DataFrame, horizon: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Partition data into proper train, validation/calibration, and test
        # If horizon is provided, enforces boundary separation on BOTH origin and target dates
        dates = pd.to_datetime(df["Date"])
        val_cutoff_date = self.get_val_cutoff_date(df)

        if horizon is not None:
            target_date_col = f"target_H{horizon}_date"
            if target_date_col in df.columns:
                target_dates = pd.to_datetime(df[target_date_col])
            else:
                target_dates = dates + pd.Timedelta(days=horizon)

            # Train: origin < val_cutoff AND target < val_cutoff
            train_mask = (
                (dates >= self.train_start)
                & (dates < val_cutoff_date)
                & (target_dates < val_cutoff_date)
            )

            # Validation: origin >= val_cutoff AND origin <= train_end AND target < test_start
            val_mask = (
                (dates >= val_cutoff_date)
                & (dates <= self.train_end)
                & (target_dates < self.test_start)
            )

            # Test: origin >= test_start AND target <= self.test_end + Timedelta(days=horizon)
            test_mask = (
                (dates >= self.test_start)
                & (dates <= self.test_end)
            )

            train_split = df[train_mask].copy()
            val_split = df[val_mask].copy()
            test_split = df[test_mask].copy()

            # Programmatic verification assertions
            if len(train_split) > 0 and len(val_split) > 0:
                t_dates = pd.to_datetime(train_split[target_date_col] if target_date_col in train_split else pd.to_datetime(train_split["Date"]) + pd.Timedelta(days=horizon))
                v_dates = pd.to_datetime(val_split["Date"])
                assert (t_dates < v_dates.min()).all(), f"H{horizon}: Training target dates overlap into validation period!"

            if len(val_split) > 0 and len(test_split) > 0:
                v_t_dates = pd.to_datetime(val_split[target_date_col] if target_date_col in val_split else pd.to_datetime(val_split["Date"]) + pd.Timedelta(days=horizon))
                ts_dates = pd.to_datetime(test_split["Date"])
                assert (v_t_dates < ts_dates.min()).all(), f"H{horizon}: Validation target dates overlap into test period!"

            # Assert disjoint origin dates
            assert len(set(train_split["Date"]).intersection(set(val_split["Date"]))) == 0, "Train and Val origin dates overlap!"
            assert len(set(val_split["Date"]).intersection(set(test_split["Date"]))) == 0, "Val and Test origin dates overlap!"
            assert len(set(train_split["Date"]).intersection(set(test_split["Date"]))) == 0, "Train and Test origin dates overlap!"

        else:
            # Origin-date-only fallback
            full_train = df[(dates >= self.train_start) & (dates <= self.train_end)].copy()
            train_split = full_train[full_train["Date"] < val_cutoff_date].copy()
            val_split = full_train[full_train["Date"] >= val_cutoff_date].copy()
            test_split = df[(dates >= self.test_start) & (dates <= self.test_end)].copy()

        return train_split, val_split, test_split

    def get_seasonal_holdout_split(
        self, df: pd.DataFrame, horizon: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Seasonal hold-out scenario:
        - Train & validation: Mar–Oct only (months 3–10 by target date).
        - Calibration: validation period slice of Mar–Oct (2018).
        - Test: Nov–Feb of 2019–2020 (months 11, 12, 1, 2 by target date).
        - Purged by target date: no training or validation targets leak into the test period.
        """
        dates = pd.to_datetime(df["Date"])
        target_date_col = f"target_H{horizon}_date"
        if target_date_col in df.columns:
            target_dates = pd.to_datetime(df[target_date_col])
        else:
            target_dates = dates + pd.Timedelta(days=horizon)

        summer_months = {3, 4, 5, 6, 7, 8, 9, 10}
        winter_months = {11, 12, 1, 2}

        val_cutoff_date = self.get_val_cutoff_date(df)

        train_mask = (
            (dates >= self.train_start)
            & (dates < val_cutoff_date)
            & (target_dates < val_cutoff_date)
            & (target_dates.dt.month.isin(summer_months))
        )
        val_mask = (
            (dates >= val_cutoff_date)
            & (dates <= self.train_end)
            & (target_dates < self.test_start)
            & (target_dates.dt.month.isin(summer_months))
        )
        test_mask = (
            (dates >= self.test_start)
            & (dates <= self.test_end)
            & (target_dates.dt.month.isin(winter_months))
            & (target_dates.dt.year.isin([2019, 2020]))
        )

        train_split = df[train_mask].copy()
        val_split = df[val_mask].copy()
        test_split = df[test_mask].copy()

        if len(train_split) > 0 and len(test_split) > 0:
            t_dates = (
                pd.to_datetime(train_split[target_date_col])
                if target_date_col in train_split
                else pd.to_datetime(train_split["Date"]) + pd.Timedelta(days=horizon)
            )
            ts_dates = pd.to_datetime(test_split["Date"])
            assert (t_dates < ts_dates.min()).all(), f"H{horizon}: Training target dates overlap test origins!"

        return train_split, val_split, test_split


class ShiftScenarioPartitioner:
    # Generates specific distribution shift evaluation slices
    @staticmethod
    def get_seasonal_slices(test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Partition test data into standard Indian seasons
        slices = {}
        for season in ["Winter", "Summer", "Monsoon", "PostMonsoon"]:
            slice_df = test_df[test_df["Season"] == season].copy()
            if len(slice_df) > 0:
                slices[f"Season_{season}"] = slice_df
        return slices

    @staticmethod
    def get_event_slices(test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Extract event windows: Diwali, stubble burning, and COVID-19 lockdown
        slices = {}
        diwali_df = test_df[test_df["is_diwali_window"] == 1].copy()
        if len(diwali_df) > 0:
            slices["Event_Diwali"] = diwali_df

        stubble_df = test_df[test_df["is_stubble_burning"] == 1].copy()
        if len(stubble_df) > 0:
            slices["Event_StubbleBurning"] = stubble_df

        lockdown_df = test_df[test_df["is_lockdown"] == 1].copy()
        if len(lockdown_df) > 0:
            slices["Event_Lockdown"] = lockdown_df

        inversion_df = test_df[test_df["is_winter_inversion"] == 1].copy()
        if len(inversion_df) > 0:
            slices["Event_WinterInversion"] = inversion_df

        return slices

    @staticmethod
    def get_year_slices(test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Split test set into 2019 and 2020 regimes
        slices = {}
        for yr in [2019, 2020]:
            slice_df = test_df[test_df["Year"] == yr].copy()
            if len(slice_df) > 0:
                slices[f"Year_{yr}"] = slice_df
        return slices

    @staticmethod
    def get_city_slices(test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Split test set by city
        slices = {}
        for city in test_df["City"].unique():
            slices[f"City_{city}"] = test_df[test_df["City"] == city].copy()
        return slices
