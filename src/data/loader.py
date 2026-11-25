# Data loader and chronological calendar validator for AQI dataset
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

TARGET_CITIES = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Lucknow"]
CORE_POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
ALL_FEATURE_POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]


def load_raw_city_data(filepath: str = "data/raw/city_day.csv", cities: Optional[List[str]] = None) -> pd.DataFrame:
    # Load raw city_day.csv dataset
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    if cities is None:
        cities = TARGET_CITIES
    df = df[df["City"].isin(cities)].copy()
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    return df


def reindex_city_calendar(df: pd.DataFrame, city: str) -> pd.DataFrame:
    # Reindex city series to a full consecutive daily calendar
    city_df = df[df["City"] == city].copy()
    city_df = city_df.sort_values("Date").reset_index(drop=True)
    min_date = city_df["Date"].min()
    max_date = city_df["Date"].max()
    full_idx = pd.date_range(min_date, max_date, freq="D", name="Date")
    city_df = city_df.set_index("Date").reindex(full_idx)
    city_df["City"] = city
    city_df = city_df.reset_index()
    # Mark target availability
    city_df["AQI_observed"] = city_df["AQI"].notna().astype(int)
    return city_df


def load_and_prepare_all_cities(filepath: str = "data/raw/city_day.csv") -> pd.DataFrame:
    # Load and reindex all target cities with verified chronological order
    raw_df = load_raw_city_data(filepath)
    city_dfs = []
    for city in TARGET_CITIES:
        city_dfs.append(reindex_city_calendar(raw_df, city))
    combined_df = pd.concat(city_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(["City", "Date"]).reset_index(drop=True)
    return combined_df
