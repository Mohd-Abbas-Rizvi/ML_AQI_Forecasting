# Controlled Ablation Experiments for AQI Research Pipeline
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from src.models.classical import ModelFactory, get_feature_columns
from src.data.splitters import TemporalSplitter
from src.evaluation.metrics import calculate_point_metrics, calculate_interval_metrics
from src.uncertainty.conformal import SplitConformalPredictor, RollingConformalPredictor


def run_ablation_imputation(results_dir: str = "results/tables"):
    # Ablation 1: Controlled comparison of 3 missing-value imputation methods
    print("\n--- Running Ablation 1: Imputation Methods Comparison ---")
    methods = ["ffill_train_median", "train_season_median", "causal_rolling_mean"]
    records = []

    for m in methods:
        fpath = f"data/processed/features_{m}.csv"
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        df["Date"] = pd.to_datetime(df["Date"])
        feature_cols = get_feature_columns(df)

        splitter = TemporalSplitter()
        tr, val, ts = splitter.get_splits(df, horizon=1)

        # For H=1
        tr_valid = tr[tr["target_H1_valid"] == 1]
        ts_valid = ts[ts["target_H1_valid"] == 1]

        X_tr, y_tr = tr_valid[feature_cols].values, tr_valid["target_H1"].values
        X_ts, y_ts = ts_valid[feature_cols].values, ts_valid["target_H1"].values

        for model_name in ["Ridge", "GradientBoosting"]:
            model = ModelFactory.create_model(model_name, random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_ts)
            pt = calculate_point_metrics(y_ts, preds)

            records.append({
                "Ablation": "Imputation",
                "Method": m,
                "Model": model_name,
                "Horizon": "H1",
                "MAE": pt["MAE"],
                "RMSE": pt["RMSE"],
                "R2": pt["R2"],
            })

    res_df = pd.DataFrame(records)
    out_path = os.path.join(results_dir, "ablation_imputation.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved imputation ablation to {out_path}")
    print(res_df.to_string())
    return res_df


def run_ablation_feature_groups(results_dir: str = "results/tables"):
    # Ablation 2, 3, 4: Feature Groupings (Lags vs Pollutants vs Calendar vs Events)
    print("\n--- Running Ablation 2-4: Feature Groupings ---")
    fpath = "data/processed/features_ffill_train_median.csv"
    df = pd.read_csv(fpath)
    df["Date"] = pd.to_datetime(df["Date"])
    all_cols = get_feature_columns(df)

    splitter = TemporalSplitter()
    tr, val, ts = splitter.get_splits(df, horizon=1)
    tr_valid = tr[tr["target_H1_valid"] == 1]
    ts_valid = ts[ts["target_H1_valid"] == 1]
    y_tr = tr_valid["target_H1"].values
    y_ts = ts_valid["target_H1"].values

    # Define subsets
    lag_cols = [c for c in all_cols if "AQI_lag" in c or "AQI_roll" in c]
    pol_cols = [c for c in all_cols if any(p in c for p in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])]
    cal_cols = [c for c in all_cols if any(k in c for k in ["Day", "Month", "Year", "sin_", "cos_", "is_weekend", "Season_code"])]
    evt_cols = [c for c in all_cols if any(k in c for k in ["diwali", "stubble", "inversion", "lockdown"])]

    subsets = {
        "All_Features": all_cols,
        "AQI_Lags_Only": lag_cols,
        "AQI_Lags_Plus_Pollutants": sorted(list(set(lag_cols + pol_cols))),
        "AQI_Lags_Plus_Calendar": sorted(list(set(lag_cols + cal_cols))),
        "No_Event_Flags": [c for c in all_cols if c not in evt_cols],
        "No_Calendar_Features": [c for c in all_cols if c not in cal_cols],
    }

    records = []
    for sname, cols in subsets.items():
        if len(cols) == 0:
            continue
        X_tr = tr_valid[cols].values
        X_ts = ts_valid[cols].values

        for model_name in ["Ridge", "GradientBoosting"]:
            model = ModelFactory.create_model(model_name, random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_ts)
            pt = calculate_point_metrics(y_ts, preds)

            records.append({
                "Feature_Configuration": sname,
                "NumFeatures": len(cols),
                "Model": model_name,
                "Horizon": "H1",
                "MAE": pt["MAE"],
                "RMSE": pt["RMSE"],
                "R2": pt["R2"],
            })

    res_df = pd.DataFrame(records)
    out_path = os.path.join(results_dir, "ablation_features.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved feature group ablation to {out_path}")
    print(res_df.to_string())
    return res_df


def run_ablation_conformal_window(results_dir: str = "results/tables"):
    # Ablation 6: Rolling conformal window sensitivity (W = 30, 60, 120, 180)
    print("\n--- Running Ablation 6: Conformal Rolling Window Sensitivity ---")
    fpath = "data/processed/features_ffill_train_median.csv"
    df = pd.read_csv(fpath)
    df["Date"] = pd.to_datetime(df["Date"])
    cols = get_feature_columns(df)

    splitter = TemporalSplitter()
    tr, val, ts = splitter.get_splits(df, horizon=1)
    tr_valid = tr[tr["target_H1_valid"] == 1]
    val_valid = val[val["target_H1_valid"] == 1]
    ts_valid = ts[ts["target_H1_valid"] == 1]

    X_tr, y_tr = tr_valid[cols].values, tr_valid["target_H1"].values
    X_val, y_val = val_valid[cols].values, val_valid["target_H1"].values
    X_ts, y_ts = ts_valid[cols].values, ts_valid["target_H1"].values

    model = ModelFactory.create_model("GradientBoosting", random_state=42)
    model.fit(X_tr, y_tr)
    val_preds = model.predict(X_val)
    ts_preds = model.predict(X_ts)
    val_scores = np.abs(y_val - val_preds)

    # Static baseline
    sp = SplitConformalPredictor(alpha=0.10)
    sp.calibrate(y_val, val_preds)
    lo_stat, up_stat = sp.predict_intervals(ts_preds)
    stat_metrics = calculate_interval_metrics(y_ts, lo_stat, up_stat, nominal=0.90)

    records = [{
        "Calibration_Type": "Static_Fixed",
        "Window_Size": 0,
        "PICP": stat_metrics["PICP"],
        "MPIW": stat_metrics["MPIW"],
        "CoverageGap": stat_metrics["CoverageGap"],
    }]

    for w in [30, 60, 90, 120, 180]:
        rcp = RollingConformalPredictor(alpha=0.10, window_size=w, horizon=1, mode="pooled")
        lo_roll, up_roll, _ = rcp.run_rolling_evaluation(ts_valid, ts_preds, y_ts, val_scores, val_valid["City"].to_numpy())
        roll_metrics = calculate_interval_metrics(y_ts, lo_roll, up_roll, nominal=0.90)

        records.append({
            "Calibration_Type": f"Rolling_Window_{w}",
            "Window_Size": w,
            "PICP": roll_metrics["PICP"],
            "MPIW": roll_metrics["MPIW"],
            "CoverageGap": roll_metrics["CoverageGap"],
        })

    res_df = pd.DataFrame(records)
    out_path = os.path.join(results_dir, "ablation_conformal_window.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved conformal window ablation to {out_path}")
    print(res_df.to_string())
    return res_df


def run_spatial_held_out_city_experiment(results_dir: str = "results/tables"):
    # Spatial Shift: Leave-one-city-out cross-validation evaluated on 2019-2020 test window with CIs
    print("\n--- Running Spatial Shift: Held-Out City Transfer (2019-2020 Window) ---")
    fpath = "data/processed/features_ffill_train_median.csv"
    df = pd.read_csv(fpath)
    df["Date"] = pd.to_datetime(df["Date"])
    cols = get_feature_columns(df)

    cities = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Lucknow"]
    records = []

    for held_out in cities:
        # Train on 4 cities (2015-2018), test on held-out city strictly on 2019-2020 test window
        train_mask = (df["City"] != held_out) & (df["Date"] <= "2018-12-31") & (df["target_H1_valid"] == 1)
        test_mask  = (df["City"] == held_out) & (df["Date"] >= "2019-01-01") & (df["Date"] <= "2020-07-01") & (df["target_H1_valid"] == 1)

        city_test = df.loc[test_mask].sort_values("Date").reset_index(drop=True)

        X_tr = df.loc[train_mask, cols].values
        y_tr = df.loc[train_mask, "target_H1"].values
        X_ts = city_test[cols].values
        y_ts = city_test["target_H1"].values

        # 1. Local Persistence baseline
        pers_preds = city_test["AQI_lag_0"].values
        pers_pt = calculate_point_metrics(y_ts, pers_preds)

        # 2. GradientBoosting zero-shot transfer model
        model = ModelFactory.create_model("GradientBoosting", random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_ts)
        pt = calculate_point_metrics(y_ts, preds)

        # 3. 14-Day Moving Block Bootstrap for paired difference vs Persistence
        e_gb   = np.abs(y_ts - preds)
        e_pers = np.abs(y_ts - pers_preds)
        n = len(y_ts)
        b_size = 14
        n_blocks = int(np.ceil(n / b_size))
        B = 1000
        rng = np.random.default_rng(42)

        e_gb_circ   = np.concatenate([e_gb, e_gb[:b_size]])
        e_pers_circ = np.concatenate([e_pers, e_pers[:b_size]])

        diffs = np.empty(B)
        pcts  = np.empty(B)

        for b in range(B):
            starts = rng.integers(0, n, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + b_size) for s in starts])[:n]
            m_g = np.mean(e_gb_circ[idx])
            m_p = np.mean(e_pers_circ[idx])
            diffs[b] = m_g - m_p
            pcts[b]  = ((m_p - m_g) / m_p * 100.0) if m_p > 0 else 0.0

        ci_diff_lo = float(np.percentile(diffs, 2.5))
        ci_diff_hi = float(np.percentile(diffs, 97.5))
        ci_pct_lo  = float(np.percentile(pcts, 2.5))
        ci_pct_hi  = float(np.percentile(pcts, 97.5))

        obs_diff = pt["MAE"] - pers_pt["MAE"]
        obs_pct  = ((pers_pt["MAE"] - pt["MAE"]) / pers_pt["MAE"] * 100.0) if pers_pt["MAE"] > 0 else 0.0

        records.append({
            "HeldOutCity": held_out,
            "Model": "Persistence",
            "TrainCities": f"Local lag-0 (No cross-city training)",
            "TestWindow": "2019-2020",
            "MAE": pers_pt["MAE"],
            "RMSE": pers_pt["RMSE"],
            "R2": pers_pt["R2"],
            "Diff_MAE": 0.0,
            "Diff_MAE_CI95_Lower": 0.0,
            "Diff_MAE_CI95_Upper": 0.0,
            "Pct_Improvement": 0.0,
            "Pct_Improvement_CI95_Lower": 0.0,
            "Pct_Improvement_CI95_Upper": 0.0,
            "Count": pers_pt["Count"],
        })

        records.append({
            "HeldOutCity": held_out,
            "Model": "GradientBoosting",
            "TrainCities": f"All except {held_out}",
            "TestWindow": "2019-2020",
            "MAE": pt["MAE"],
            "RMSE": pt["RMSE"],
            "R2": pt["R2"],
            "Diff_MAE": obs_diff,
            "Diff_MAE_CI95_Lower": ci_diff_lo,
            "Diff_MAE_CI95_Upper": ci_diff_hi,
            "Pct_Improvement": obs_pct,
            "Pct_Improvement_CI95_Lower": ci_pct_lo,
            "Pct_Improvement_CI95_Upper": ci_pct_hi,
            "Count": pt["Count"],
        })

    res_df = pd.DataFrame(records)
    out_path = os.path.join(results_dir, "spatial_held_out_city.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved spatial transfer results (2019-2020 window) to {out_path}")
    print(res_df.to_string())
    return res_df


if __name__ == "__main__":
    run_ablation_imputation()
    run_ablation_feature_groups()
    run_ablation_conformal_window()
    run_spatial_held_out_city_experiment()
