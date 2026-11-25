# Preprocessing and dataset preparation pipeline
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.data.loader import load_and_prepare_all_cities, TARGET_CITIES
from src.data.imputation import LeakageSafeImputer
from src.features.engineer import build_full_feature_pipeline

POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"]


def run_preprocessing_pipeline(
    raw_path: str = "data/raw/city_day.csv",
    output_dir: str = "data/processed",
    results_dir: str = "results/tables",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Step 1: Loading raw city data and enforcing strict daily calendar continuity...")
    raw_df = load_and_prepare_all_cities(raw_path)
    print(f"Total calendar records across 5 cities: {len(raw_df)}")

    # Profiling missingness
    profile_rows = []
    for city in TARGET_CITIES:
        c_raw = raw_df[raw_df["City"] == city]
        n_days = len(c_raw)
        aqi_avail = c_raw["AQI"].notna().sum()
        profile_rows.append({
            "City": city,
            "TotalCalendarDays": n_days,
            "StartDate": c_raw["Date"].min().strftime("%Y-%m-%d"),
            "EndDate": c_raw["Date"].max().strftime("%Y-%m-%d"),
            "AQI_Available": aqi_avail,
            "AQI_Missing": n_days - aqi_avail,
            "AQI_MissingPct": round((n_days - aqi_avail) / n_days * 100, 2),
            "PM2.5_MissingPct": round(c_raw["PM2.5"].isnull().mean() * 100, 2),
            "NO2_MissingPct": round(c_raw["NO2"].isnull().mean() * 100, 2),
        })

    profile_df = pd.DataFrame(profile_rows)
    profile_path = os.path.join(results_dir, "city_data_profile.csv")
    profile_df.to_csv(profile_path, index=False)
    print(f"Saved dataset profile to {profile_path}")
    print(profile_df.to_string())

    imputation_methods = ["ffill_train_median", "train_season_median", "causal_rolling_mean"]
    processed_dfs = {}

    for method in imputation_methods:
        print(f"\nStep 2: Processing with Imputation Method: {method} (fitted strictly on proper train: Date <= 2018-03-13)...")
        imputer = LeakageSafeImputer(method=method, train_end_date="2018-03-13")
        imputer.fit(raw_df, POLLUTANT_COLS)

        full_features = build_full_feature_pipeline(raw_df, imputer)

        # Save processed dataset
        out_path = os.path.join(output_dir, f"features_{method}.csv")
        full_features.to_csv(out_path, index=False)
        print(f"Saved {method} features to {out_path} ({full_features.shape[0]} rows, {full_features.shape[1]} cols)")
        processed_dfs[method] = full_features

    # Report valid sample counts per city and horizon
    primary_df = processed_dfs["ffill_train_median"]
    sample_counts = []
    for city in TARGET_CITIES:
        cdf = primary_df[primary_df["City"] == city]
        train_mask = (cdf["Date"] >= "2015-01-01") & (cdf["Date"] <= "2018-12-31")
        test_mask = (cdf["Date"] >= "2019-01-01") & (cdf["Date"] <= "2020-07-01")

        h1_train_valid = (train_mask & (cdf["target_H1_valid"] == 1)).sum()
        h1_test_valid = (test_mask & (cdf["target_H1_valid"] == 1)).sum()
        h3_train_valid = (train_mask & (cdf["target_H3_valid"] == 1)).sum()
        h3_test_valid = (test_mask & (cdf["target_H3_valid"] == 1)).sum()

        sample_counts.append({
            "City": city,
            "TotalDays": len(cdf),
            "Train_Days": train_mask.sum(),
            "Test_Days": test_mask.sum(),
            "H1_Train_Valid": h1_train_valid,
            "H1_Test_Valid": h1_test_valid,
            "H3_Train_Valid": h3_train_valid,
            "H3_Test_Valid": h3_test_valid,
        })

    counts_df = pd.DataFrame(sample_counts)
    counts_path = os.path.join(results_dir, "sample_counts_by_horizon.csv")
    counts_df.to_csv(counts_path, index=False)
    print(f"\nSaved valid sample counts to {counts_path}")
    print(counts_df.to_string())

    print("\nPhase 3 Preprocessing Successfully Completed.")
    return processed_dfs


if __name__ == "__main__":
    run_preprocessing_pipeline()
