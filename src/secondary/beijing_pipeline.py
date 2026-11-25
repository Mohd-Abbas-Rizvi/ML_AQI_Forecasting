# Secondary Experiment: UCI Beijing PM2.5 Dataset Pipeline (2010-2014)
# Follows authoritative poster protocol: Train 2010-2012, Shifted Test 2013-2014
import os
import urllib.request
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

from src.models.classical import ModelFactory
from src.evaluation.metrics import calculate_point_metrics, calculate_interval_metrics, calculate_shift_penalty
from src.uncertainty.conformal import SplitConformalPredictor, RollingConformalPredictor

UCI_BEIJING_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
RAW_BEIJING_PATH = "data/raw/PRSA_data_2010.1.1-2014.12.31.csv"
PROCESSED_BEIJING_PATH = "data/processed/beijing_daily_features.csv"


def download_beijing_data(target_path: str = RAW_BEIJING_PATH) -> bool:
    # Attempt to download the official UCI Beijing PM2.5 dataset if network is available
    if os.path.exists(target_path):
        print(f"Beijing raw data already exists at {target_path}")
        return True

    print(f"Downloading UCI Beijing PM2.5 dataset from {UCI_BEIJING_URL}...")
    try:
        urllib.request.urlretrieve(UCI_BEIJING_URL, target_path)
        print(f"Successfully downloaded to {target_path}")
        return True
    except Exception as e:
        print(f"Could not download UCI Beijing dataset automatically: {e}")
        print("To run the Beijing study manually:")
        print(f"1. Download the CSV from: {UCI_BEIJING_URL}")
        print(f"2. Place the file at: {target_path}")
        return False


def load_and_preprocess_beijing(raw_path: str = RAW_BEIJING_PATH) -> Optional[pd.DataFrame]:
    # Aggregate 43,824 hourly observations to daily means and build lag features
    if not os.path.exists(raw_path):
        print(f"Beijing dataset file not found at {raw_path}.")
        return None

    df = pd.read_csv(raw_path)
    print(f"Loaded raw Beijing hourly dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Construct datetime column
    df["Date"] = pd.to_datetime(df[["year", "month", "day"]])

    # Missing value rate check
    missing_pm25 = df["pm2.5"].isnull().sum()
    missing_pct = missing_pm25 / len(df) * 100
    print(f"Hourly PM2.5 missing: {missing_pm25} / {len(df)} ({missing_pct:.2f}%) [Poster verifies ~4.7%]")

    # Aggregate to daily observations
    daily = df.groupby("Date").agg({
        "pm2.5": "mean",
        "DEWP": "mean",
        "TEMP": "mean",
        "PRES": "mean",
        "Iws": "mean",
        "Is": "sum",
        "Ir": "sum",
    }).reset_index()

    print(f"Daily aggregated observations: {len(daily)} days (2010-01-01 to 2014-12-31)")

    # Impute missing daily PM2.5 values strictly causal forward-fill + train median fallback
    daily = daily.sort_values("Date").reset_index(drop=True)
    train_mask = daily["Date"] <= "2012-12-31"
    train_pm25_med = daily.loc[train_mask, "pm2.5"].median()

    daily["pm2.5_raw"] = daily["pm2.5"].copy()
    daily["pm2.5_imputed"] = daily["pm2.5"].ffill().fillna(train_pm25_med)

    # Lags and rolling features
    for lag in [1, 2, 3, 7, 14, 30]:
        daily[f"pm2.5_lag_{lag}"] = daily["pm2.5_imputed"].shift(lag)
    for w in [7, 14, 30]:
        daily[f"pm2.5_roll_mean_{w}"] = daily["pm2.5_imputed"].shift(1).rolling(w, min_periods=1).mean()
        daily[f"pm2.5_roll_std_{w}"] = daily["pm2.5_imputed"].shift(1).rolling(w, min_periods=1).std().fillna(0.0)

    # Weather lags
    for col in ["DEWP", "TEMP", "PRES", "Iws"]:
        daily[f"{col}_lag_1"] = daily[col].shift(1)
        daily[f"{col}_roll_mean_7"] = daily[col].shift(1).rolling(7, min_periods=1).mean()

    # Calendar features
    daily["Month"] = daily["Date"].dt.month
    daily["DayOfWeek"] = daily["Date"].dt.day_of_week
    daily["DayOfYear"] = daily["Date"].dt.day_of_year
    daily["sin_DayOfYear"] = np.sin(2 * np.pi * daily["DayOfYear"] / 365.25)
    daily["cos_DayOfYear"] = np.cos(2 * np.pi * daily["DayOfYear"] / 365.25)

    # Targets: H=1 and H=3 from raw unimputed PM2.5
    daily["target_H1"] = daily["pm2.5_raw"].shift(-1)
    daily["target_H3"] = daily["pm2.5_raw"].shift(-3)
    daily["target_H1_valid"] = daily["target_H1"].notna().astype(int)
    daily["target_H3_valid"] = daily["target_H3"].notna().astype(int)

    os.makedirs(os.path.dirname(PROCESSED_BEIJING_PATH), exist_ok=True)
    daily.to_csv(PROCESSED_BEIJING_PATH, index=False)
    print(f"Saved processed Beijing daily features to {PROCESSED_BEIJING_PATH}")
    return daily


def run_beijing_shift_experiment(results_dir: str = "results/tables"):
    # Run the 2010-2012 training vs 2013-2014 shifted test evaluation
    daily = load_and_preprocess_beijing()
    if daily is None:
        print("Skipping Beijing experiment execution (dataset not present).")
        return None

    exclude = [
        "Date", "pm2.5_raw", "target_H1", "target_H3", "target_H1_valid", "target_H3_valid"
    ]
    feature_cols = [c for c in daily.columns if c not in exclude and daily[c].dtype in [float, int, np.float64, np.int64]]

    # Train: 2010-2012, Test: 2013-2014
    train_mask = (daily["Date"] >= "2010-01-01") & (daily["Date"] <= "2012-12-31")
    test_mask = (daily["Date"] >= "2013-01-01") & (daily["Date"] <= "2014-12-31")

    # Validation: last 20% of train
    train_dates = daily.loc[train_mask, "Date"].unique()
    val_cutoff = train_dates[int(len(train_dates) * 0.8)]

    tr_sub = train_mask & (daily["Date"] < val_cutoff)
    val_sub = train_mask & (daily["Date"] >= val_cutoff)

    records = []
    for horizon in [1, 3]:
        target_col = f"target_H{horizon}"
        valid_col = f"target_H{horizon}_valid"

        tr_v = daily[tr_sub & (daily[valid_col] == 1)]
        val_v = daily[val_sub & (daily[valid_col] == 1)]
        ts_v = daily[test_mask & (daily[valid_col] == 1)]

        X_tr, y_tr = tr_v[feature_cols].values, tr_v[target_col].values
        X_val, y_val = val_v[feature_cols].values, val_v[target_col].values
        X_ts, y_ts = ts_v[feature_cols].values, ts_v[target_col].values

        # Reference Random Split (same size as test set)
        np.random.seed(42)
        all_v = daily[daily[valid_col] == 1]
        rand_idx = np.random.choice(len(all_v), size=len(ts_v), replace=False)
        rand_v = all_v.iloc[rand_idx]
        X_rand, y_rand = rand_v[feature_cols].values, rand_v[target_col].values

        models = {
            "Persistence": None,
            "Ridge": ModelFactory.create_model("Ridge", random_state=42),
            "GradientBoosting": ModelFactory.create_model("GradientBoosting", random_state=42),
            "RandomForest": ModelFactory.create_model("RandomForest", random_state=42),
        }

        for mname, model in models.items():
            if mname == "Persistence":
                p_rand = rand_v["pm2.5_imputed"].values
                p_ts = ts_v["pm2.5_imputed"].values
                p_val = val_v["pm2.5_imputed"].values
            else:
                model.fit(X_tr, y_tr)
                p_rand = model.predict(X_rand)
                p_ts = model.predict(X_ts)
                p_val = model.predict(X_val)

            ref_pt = calculate_point_metrics(y_rand, p_rand)
            ts_pt = calculate_point_metrics(y_ts, p_ts)
            shift_pen = calculate_shift_penalty(ts_pt["MAE"], ref_pt["MAE"])

            # Conformal
            sp = SplitConformalPredictor(alpha=0.10)
            sp.calibrate(y_val, p_val)
            lo_s, up_s = sp.predict_intervals(p_ts)
            conf_s = calculate_interval_metrics(y_ts, lo_s, up_s, nominal=0.90)

            # Rolling Conformal
            val_scores = np.abs(y_val - p_val)
            rcp = RollingConformalPredictor(alpha=0.10, window_size=120, horizon=horizon, mode="pooled")
            lo_r, up_r, _ = rcp.run_rolling_evaluation(ts_v, p_ts, y_ts, val_scores)
            conf_r = calculate_interval_metrics(y_ts, lo_r, up_r, nominal=0.90)

            records.append({
                "Dataset": "UCI_Beijing_PM2.5",
                "Horizon": f"H{horizon}",
                "Model": mname,
                "MAE_Test_2013_2014": ts_pt["MAE"],
                "RMSE_Test_2013_2014": ts_pt["RMSE"],
                "R2_Test_2013_2014": ts_pt["R2"],
                "ShiftPenalty": shift_pen,
                "PICP_Static": conf_s["PICP"],
                "MPIW_Static": conf_s["MPIW"],
                "PICP_Rolling": conf_r["PICP"],
                "MPIW_Rolling": conf_r["MPIW"],
            })

    res_df = pd.DataFrame(records)
    out_table = os.path.join(results_dir, "beijing_shift_results.csv")
    res_df.to_csv(out_table, index=False)
    print(f"Saved Beijing PM2.5 shift benchmark results to {out_table}")
    print(res_df.to_string())
    return res_df


if __name__ == "__main__":
    download_beijing_data()
    run_beijing_shift_experiment()
