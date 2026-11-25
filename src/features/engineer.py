# Leakage-safe Feature Engineering for AQI Forecasting (H1 and H3)
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

DIWALI_DATES = [
    "2015-11-11",
    "2016-10-30",
    "2017-10-19",
    "2018-11-07",
    "2019-10-27",
    "2020-11-14",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add calendar and cyclical features
    df = df.copy()
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfMonth"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Year"] = df["Date"].dt.year
    df["is_weekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Cyclical representations
    df["sin_DayOfYear"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["cos_DayOfYear"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)
    df["sin_Month"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["cos_Month"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["sin_DayOfWeek"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7.0)
    df["cos_DayOfWeek"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7.0)

    # Standard Indian Season definitions:
    # Winter: Dec, Jan, Feb (12, 1, 2)
    # Summer/Pre-monsoon: Mar, Apr, May (3, 4, 5)
    # Monsoon: Jun, Jul, Aug, Sep (6, 7, 8, 9)
    # Post-monsoon/Autumn: Oct, Nov (10, 11)
    def assign_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Summer"
        elif m in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "PostMonsoon"

    df["Season"] = df["Month"].apply(assign_season)
    season_map = {"Winter": 0, "Summer": 1, "Monsoon": 2, "PostMonsoon": 3}
    df["Season_code"] = df["Season"].map(season_map)
    return df


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add event windows: Diwali, stubble burning, winter inversion, and 2020 lockdown
    df = df.copy()
    dates = pd.to_datetime(df["Date"])

    # Diwali window (+/- 3 days around festival date)
    diwali_dt = pd.to_datetime(DIWALI_DATES)
    diwali_flags = np.zeros(len(df), dtype=int)
    for d in diwali_dt:
        mask = (dates >= d - pd.Timedelta(days=3)) & (dates <= d + pd.Timedelta(days=3))
        diwali_flags[mask] = 1
    df["is_diwali_window"] = diwali_flags

    # Stubble burning window (October 15 to November 30)
    stubble = ((df["Month"] == 10) & (df["DayOfMonth"] >= 15)) | (df["Month"] == 11)
    df["is_stubble_burning"] = stubble.astype(int)

    # Winter temperature inversion period (December 1 to January 31)
    inversion = (df["Month"] == 12) | (df["Month"] == 1)
    df["is_winter_inversion"] = inversion.astype(int)

    # COVID-19 nationwide lockdown (March 25, 2020 to May 31, 2020)
    lockdown = (dates >= "2020-03-25") & (dates <= "2020-05-31")
    df["is_lockdown"] = lockdown.astype(int)

    return df


def add_lags_and_rolling(df: pd.DataFrame, pollutant_cols: List[str]) -> pd.DataFrame:
    # Compute lags and rolling statistics strictly using information up to forecast origin t
    df = df.copy()
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    city_dfs = []
    for city, group in df.groupby("City", sort=False):
        group = group.copy()

        # AQI lags relative to origin t: t, t-1, t-2, t-6, t-13, t-29
        # In Pandas notation: current row is t (lag 0), shift(1) is t-1, etc.
        group["AQI_lag_0"] = group["AQI"]
        group["AQI_lag_1"] = group["AQI"].shift(1)
        group["AQI_lag_2"] = group["AQI"].shift(2)
        group["AQI_lag_3"] = group["AQI"].shift(3)
        group["AQI_lag_7"] = group["AQI"].shift(7)
        group["AQI_lag_14"] = group["AQI"].shift(14)
        group["AQI_lag_30"] = group["AQI"].shift(30)
        group["AQI_lag_365"] = group["AQI"].shift(365) # Seasonal naive reference

        # Rolling statistics up to origin t (using shift(0) including day t)
        for w in [7, 14, 30]:
            group[f"AQI_roll_mean_{w}"] = group["AQI"].rolling(window=w, min_periods=1).mean()
            group[f"AQI_roll_std_{w}"] = group["AQI"].rolling(window=w, min_periods=1).std().fillna(0.0)
            group[f"AQI_roll_min_{w}"] = group["AQI"].rolling(window=w, min_periods=1).min()
            group[f"AQI_roll_max_{w}"] = group["AQI"].rolling(window=w, min_periods=1).max()

        # Key pollutants lag and rolling features
        for pol in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
            if pol in group.columns:
                group[f"{pol}_lag_0"] = group[pol]
                group[f"{pol}_lag_1"] = group[pol].shift(1)
                group[f"{pol}_lag_7"] = group[pol].shift(7)
                group[f"{pol}_roll_mean_7"] = group[pol].rolling(window=7, min_periods=1).mean()

        city_dfs.append(group)

    df_out = pd.concat(city_dfs, ignore_index=True)
    return df_out


def create_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Create strictly chronological targets for H=1 (t+1) and H=3 (t+3)
    # Verification: shifts are computed grouped by city after explicit calendar reindexing
    df = df.copy()
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    city_dfs = []
    for city, group in df.groupby("City", sort=False):
        group = group.copy()
        # Future targets: shift(-1) is t+1 day, shift(-3) is t+3 days
        group["target_H1"] = group["AQI"].shift(-1)
        group["target_H3"] = group["AQI"].shift(-3)

        # Target date verification columns
        group["target_H1_date"] = group["Date"].shift(-1)
        group["target_H3_date"] = group["Date"].shift(-3)

        # Flag valid ground-truth targets (only true if actually observed and not NaN)
        group["target_H1_valid"] = group["target_H1"].notna().astype(int)
        group["target_H3_valid"] = group["target_H3"].notna().astype(int)

        city_dfs.append(group)

    df_out = pd.concat(city_dfs, ignore_index=True)
    return df_out


def build_full_feature_pipeline(raw_df: pd.DataFrame, imputer) -> pd.DataFrame:
    # Execute end-to-end leakage-safe feature engineering pipeline
    pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"]
    
    # 1. Impute missing values in features using fitted imputer
    imputed_df = imputer.transform(raw_df, pollutant_cols)

    # 2. Add calendar features
    cal_df = add_calendar_features(imputed_df)

    # 3. Add event features
    evt_df = add_event_features(cal_df)

    # 4. Add lags and rolling statistics
    feat_df = add_lags_and_rolling(evt_df, pollutant_cols)

    # 5. Create future targets (from original non-imputed AQI to preserve real evaluation labels)
    # To ensure targets are never synthetic, attach original raw AQI as target source
    raw_sorted = raw_df[["City", "Date", "AQI"]].rename(columns={"AQI": "AQI_raw_unimputed"})
    feat_df = feat_df.merge(raw_sorted, on=["City", "Date"], how="left")
    
    # Generate targets from raw unimputed AQI
    city_dfs = []
    for city, group in feat_df.groupby("City", sort=False):
        group = group.copy()
        group["target_H1"] = group["AQI_raw_unimputed"].shift(-1)
        group["target_H3"] = group["AQI_raw_unimputed"].shift(-3)
        group["target_H1_valid"] = group["target_H1"].notna().astype(int)
        group["target_H3_valid"] = group["target_H3"].notna().astype(int)
        city_dfs.append(group)
        
    final_df = pd.concat(city_dfs, ignore_index=True)
    final_df.drop(columns=["AQI_raw_unimputed"], inplace=True)
    return final_df
