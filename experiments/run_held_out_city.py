"""
Fix 2: Per-City + Persistence in Held-Out-City Table
======================================================
Adds Persistence baseline to spatial_held_out_city.csv and reports
per-city breakdown with Persistence included as a baseline row.

No model retraining needed: Persistence is computed analytically from AQI_lag_0.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.splitters import TemporalSplitter
from src.models.baselines import PersistenceForecaster
from src.models.classical import get_feature_columns
from src.evaluation.metrics import calculate_point_metrics


def run_held_out_city_with_persistence(
    features_path="data/processed/features_ffill_train_median.csv",
    results_dir="results/tables"
):
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    feature_cols = get_feature_columns(df)

    existing_path = os.path.join(results_dir, "spatial_held_out_city.csv")
    existing = pd.read_csv(existing_path) if os.path.exists(existing_path) else pd.DataFrame()

    splitter = TemporalSplitter()
    cities = sorted(df["City"].unique())
    new_records = []

    # We evaluate Persistence on H1 (the existing table uses GradientBoosting on H1 by default)
    # Also produce H3 for completeness
    for horizon in [1, 3]:
        target_col = f"target_H{horizon}"
        valid_col  = f"target_H{horizon}_valid"

        _, _, test_split = splitter.get_splits(df, horizon=horizon)
        ts_valid = test_split[test_split[valid_col] == 1].copy()

        for held_out_city in cities:
            # Test = only the held-out city test rows
            city_test = ts_valid[ts_valid["City"] == held_out_city].copy()
            if len(city_test) == 0:
                continue

            y_true = city_test[target_col].to_numpy(dtype=np.float64)

            # Persistence: predict y_{t+h} = AQI at forecast origin t
            pers = PersistenceForecaster()
            pers.fit(city_test, city_test[target_col])
            preds = pers.predict(city_test)

            pt = calculate_point_metrics(y_true, preds)

            new_records.append({
                "Horizon": f"H{horizon}",
                "HeldOutCity": held_out_city,
                "Model": "Persistence",
                "TrainCities": f"All except {held_out_city}",
                "MAE": pt["MAE"],
                "RMSE": pt["RMSE"],
                "R2": pt["R2"],
                "Count": pt["Count"],
            })

    new_df = pd.DataFrame(new_records)

    # Merge with existing table
    # Existing table has columns: HeldOutCity, Model, TrainCities, MAE, RMSE, R2, Count
    # Add Horizon column if missing (assume H1 for the existing GBM rows)
    if len(existing) > 0 and "Horizon" not in existing.columns:
        existing["Horizon"] = "H1"

    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
        subset=["Horizon", "HeldOutCity", "Model"], keep="last"
    )
    combined = combined.sort_values(["Horizon", "HeldOutCity", "Model"]).reset_index(drop=True)

    out_path = os.path.join(results_dir, "spatial_held_out_city.csv")
    combined.to_csv(out_path, index=False)
    print(f"Updated spatial held-out-city table saved to {out_path}")
    print(combined.to_string(index=False))
    return combined


if __name__ == "__main__":
    run_held_out_city_with_persistence()
