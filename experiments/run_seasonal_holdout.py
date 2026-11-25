"""
Fix 1: Held-Out-Season Experiment with Per-City Conformal Calibration
=====================================================================
Protocol:
- Train and validate on Mar–Oct only (months 3–10 by TARGET date).
- Test on Nov–Feb (months 11, 12, 1, 2) of 2019–2020 by TARGET date.
- Purged by target date: no training/cal targets leak into the test period.
- Conformal Calibration: PER-CITY
    * Static Conformal: SplitConformalPredictor fitted per-city on that city's calibration residuals.
    * Rolling Conformal: RollingConformalPredictor with mode="city_specific" and initial_cal_cities.
- Evaluation against the 85–95% nominal rule:
    * Nominal coverage = 90% (alpha = 0.10)
    * Rule: Coverage in [0.85, 0.95] -> PASS, otherwise FAIL.
    * Explicitly reports Delhi and Lucknow coverage, width, and Winkler scores.

Saves:
  results/tables/seasonal_holdout_conformal.csv
"""
import os, sys, warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.splitters import TemporalSplitter
from src.models.baselines import PersistenceForecaster
from src.models.classical import ModelFactory, get_feature_columns
from src.uncertainty.conformal import SplitConformalPredictor, RollingConformalPredictor
from src.evaluation.metrics import calculate_point_metrics, calculate_interval_metrics

ALPHA = 0.10          # 1 - nominal = 10%
NOMINAL = 1.0 - ALPHA # 0.90
RULE_MIN = 0.85
RULE_MAX = 0.95


def winkler_score(y_true, lo, up, alpha=ALPHA):
    """Winkler score: interval width + miscoverage penalty."""
    width = up - lo
    pen_lo = np.maximum(lo - y_true, 0.0)
    pen_hi = np.maximum(y_true - up, 0.0)
    penalty = (2.0 / alpha) * (pen_lo + pen_hi)
    return float(np.mean(width + penalty))


def check_rule_pass(coverage):
    """Evaluates coverage against the two-sided 85-95% rule."""
    if np.isnan(coverage):
        return "N/A"
    if coverage < RULE_MIN:
        return "FAIL"
    elif coverage > RULE_MAX:
        return "Conservative"
    else:
        return "PASS"


def block_bootstrap_coverage_ci(covered_array, block_length=14, n_bootstraps=1000, seed=42):
    """Computes moving-block bootstrap 95% CI for coverage (PICP)."""
    n = len(covered_array)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    boot_means = np.zeros(n_bootstraps)
    for b in range(n_bootstraps):
        start_indices = rng.integers(0, max(1, n - block_length + 1), size=n_blocks)
        sample = np.concatenate([covered_array[idx:idx + block_length] for idx in start_indices])[:n]
        boot_means[b] = np.mean(sample)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def run_seasonal_holdout_per_city(
    features_path="data/processed/features_ffill_train_median.csv",
    results_dir="results/tables"
):
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    feature_cols = get_feature_columns(df)
    sp = TemporalSplitter()

    records = []

    for horizon in [1, 3]:
        target_col = f"target_H{horizon}"
        valid_col  = f"target_H{horizon}_valid"

        train_split, cal_split, test_split = sp.get_seasonal_holdout_split(df, horizon=horizon)

        train_df = train_split[train_split[valid_col] == 1].copy()
        cal_df   = cal_split[cal_split[valid_col] == 1].copy()
        test_df  = test_split[test_split[valid_col] == 1].copy()

        # Sort test_df chronologically by date
        test_df = test_df.sort_values("Date").reset_index(drop=True)

        print(f"\n=======================================================", flush=True)
        print(f"  H{horizon} SEASONAL HOLDOUT (Per-City Calibration)", flush=True)
        print(f"  Train={len(train_df)}, Cal={len(cal_df)}, Test={len(test_df)}", flush=True)
        print(f"=======================================================", flush=True)

        X_tr  = train_df[feature_cols].to_numpy(dtype=np.float64)
        y_tr  = train_df[target_col].to_numpy(dtype=np.float64)
        X_cal = cal_df[feature_cols].to_numpy(dtype=np.float64)
        y_cal = cal_df[target_col].to_numpy(dtype=np.float64)
        X_ts  = test_df[feature_cols].to_numpy(dtype=np.float64)
        y_ts  = test_df[target_col].to_numpy(dtype=np.float64)

        cities = sorted(test_df["City"].unique())

        # Models to evaluate
        # 1. Persistence baseline
        pers = PersistenceForecaster()
        pers.fit(train_df, train_df[target_col])
        pers_preds_cal = pers.predict(cal_df)
        pers_preds_ts  = pers.predict(test_df)

        # 2. Validation-Selected Model: Ridge Regression
        ridge = ModelFactory.create_model("Ridge", random_state=42)
        ridge.fit(X_tr, y_tr)
        ridge_preds_cal = ridge.predict(X_cal)
        ridge_preds_ts  = ridge.predict(X_ts)

        # 3. Gradient Boosting (non-linear tree benchmark)
        gb = ModelFactory.create_model("GradientBoosting", random_state=42)
        gb.fit(X_tr, y_tr)
        gb_preds_cal = gb.predict(X_cal)
        gb_preds_ts  = gb.predict(X_ts)

        for model_name, preds_cal, preds_ts in [
            ("Persistence", pers_preds_cal, pers_preds_ts),
            ("Ridge", ridge_preds_cal, ridge_preds_ts),
            ("GradientBoosting", gb_preds_cal, gb_preds_ts),
        ]:
            cal_residuals = np.abs(y_cal - preds_cal)

            # --- A. Per-City Static Conformal Calibration ---
            lo_static = np.zeros(len(test_df))
            up_static = np.zeros(len(test_df))

            city_sp_predictors = {}
            for c in cities:
                c_mask_cal = (cal_df["City"] == c).to_numpy()
                c_mask_ts  = (test_df["City"] == c).to_numpy()

                sp_c = SplitConformalPredictor(alpha=ALPHA)
                sp_c.calibrate(y_cal[c_mask_cal], preds_cal[c_mask_cal])
                city_sp_predictors[c] = sp_c

                c_preds_ts = preds_ts[c_mask_ts]
                c_lo, c_up = sp_c.predict_intervals(c_preds_ts)
                lo_static[c_mask_ts] = c_lo
                up_static[c_mask_ts] = c_up

            # --- B. Per-City Rolling Conformal Calibration ---
            # window_size=30 was the default for the held-out winter (window spans ~179 days)
            roll_conf = RollingConformalPredictor(
                alpha=ALPHA, window_size=30, horizon=horizon, mode="city_specific"
            )
            lo_roll, up_roll, _ = roll_conf.run_rolling_evaluation(
                test_df=test_df,
                predictions=preds_ts,
                y_true=y_ts,
                initial_cal_scores=cal_residuals,
                initial_cal_cities=cal_df["City"].to_numpy()
            )

            # --- C. Pooled Metrics & Coverage Bootstrap CIs ---
            pt_pooled = calculate_point_metrics(y_ts, preds_ts)
            int_s_pool = calculate_interval_metrics(y_ts, lo_static, up_static, nominal=NOMINAL)
            int_r_pool = calculate_interval_metrics(y_ts, lo_roll, up_roll, nominal=NOMINAL)
            w_s_pool   = winkler_score(y_ts, lo_static, up_static)
            w_r_pool   = winkler_score(y_ts, lo_roll, up_roll)

            cov_s_pool_arr = (y_ts >= lo_static) & (y_ts <= up_static)
            cov_r_pool_arr = (y_ts >= lo_roll) & (y_ts <= up_roll)
            s_ci_lo_pool, s_ci_up_pool = block_bootstrap_coverage_ci(cov_s_pool_arr)
            r_ci_lo_pool, r_ci_up_pool = block_bootstrap_coverage_ci(cov_r_pool_arr)

            records.append({
                "Horizon": f"H{horizon}",
                "City": "Pooled",
                "Model": model_name,
                "MAE": pt_pooled["MAE"],
                "RMSE": pt_pooled["RMSE"],
                "R2": pt_pooled["R2"],
                "Count": pt_pooled["Count"],
                "PICP_Static": int_s_pool["PICP"],
                "PICP_Static_CI95_Lower": s_ci_lo_pool,
                "PICP_Static_CI95_Upper": s_ci_up_pool,
                "MPIW_Static": int_s_pool["MPIW"],
                "CoverageGap_Static": int_s_pool["CoverageGap"],
                "Winkler_Static": w_s_pool,
                "PassRule_Static": check_rule_pass(int_s_pool["PICP"]),
                "PICP_Rolling": int_r_pool["PICP"],
                "PICP_Rolling_CI95_Lower": r_ci_lo_pool,
                "PICP_Rolling_CI95_Upper": r_ci_up_pool,
                "MPIW_Rolling": int_r_pool["MPIW"],
                "CoverageGap_Rolling": int_r_pool["CoverageGap"],
                "Winkler_Rolling": w_r_pool,
                "PassRule_Rolling": check_rule_pass(int_r_pool["PICP"]),
            })

            # --- D. Per-City Metrics & Coverage Bootstrap CIs ---
            for c in cities:
                c_mask = (test_df["City"] == c).to_numpy()
                y_c = y_ts[c_mask]
                p_c = preds_ts[c_mask]
                lo_s_c = lo_static[c_mask]
                up_s_c = up_static[c_mask]
                lo_r_c = lo_roll[c_mask]
                up_r_c = up_roll[c_mask]

                pt_c = calculate_point_metrics(y_c, p_c)
                int_s_c = calculate_interval_metrics(y_c, lo_s_c, up_s_c, nominal=NOMINAL)
                int_r_c = calculate_interval_metrics(y_c, lo_r_c, up_r_c, nominal=NOMINAL)
                w_s_c = winkler_score(y_c, lo_s_c, up_s_c)
                w_r_c = winkler_score(y_c, lo_r_c, up_r_c)

                cov_s_c_arr = (y_c >= lo_s_c) & (y_c <= up_s_c)
                cov_r_c_arr = (y_c >= lo_r_c) & (y_c <= up_r_c)
                s_ci_lo_c, s_ci_up_c = block_bootstrap_coverage_ci(cov_s_c_arr)
                r_ci_lo_c, r_ci_up_c = block_bootstrap_coverage_ci(cov_r_c_arr)

                records.append({
                    "Horizon": f"H{horizon}",
                    "City": c,
                    "Model": model_name,
                    "MAE": pt_c["MAE"],
                    "RMSE": pt_c["RMSE"],
                    "R2": pt_c["R2"],
                    "Count": pt_c["Count"],
                    "PICP_Static": int_s_c["PICP"],
                    "PICP_Static_CI95_Lower": s_ci_lo_c,
                    "PICP_Static_CI95_Upper": s_ci_up_c,
                    "MPIW_Static": int_s_c["MPIW"],
                    "CoverageGap_Static": int_s_c["CoverageGap"],
                    "Winkler_Static": w_s_c,
                    "PassRule_Static": check_rule_pass(int_s_c["PICP"]),
                    "PICP_Rolling": int_r_c["PICP"],
                    "PICP_Rolling_CI95_Lower": r_ci_lo_c,
                    "PICP_Rolling_CI95_Upper": r_ci_up_c,
                    "MPIW_Rolling": int_r_c["MPIW"],
                    "CoverageGap_Rolling": int_r_c["CoverageGap"],
                    "Winkler_Rolling": w_r_c,
                    "PassRule_Rolling": check_rule_pass(int_r_c["PICP"]),
                })

    res_df = pd.DataFrame(records)
    out_path = os.path.join(results_dir, "seasonal_holdout_conformal.csv")
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved per-city seasonal holdout results to {out_path}", flush=True)

    print("\n--- DELHI & LUCKNOW COVERAGE vs 85-95% RULE ---", flush=True)
    delhi_lucknow = res_df[res_df["City"].isin(["Delhi", "Lucknow"])][
        ["Horizon", "City", "Model", "PICP_Static", "PassRule_Static", "PICP_Rolling", "PassRule_Rolling", "Winkler_Static", "Winkler_Rolling"]
    ]
    print(delhi_lucknow.to_string(index=False), flush=True)

    return res_df


def run_seasonal_holdout_window_ablation(
    features_path="data/processed/features_ffill_train_median.csv",
    results_dir="results/tables",
    windows=(30, 60, 120)
):
    """
    Ablation of rolling conformal window sizes (W in [30, 60, 120])
    under the held-out winter season shift scenario (~179 calendar days).
    """
    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    feature_cols = get_feature_columns(df)
    sp = TemporalSplitter()

    records = []
    for horizon in [1, 3]:
        target_col = f"target_H{horizon}"
        valid_col  = f"target_H{horizon}_valid"

        train_split, cal_split, test_split = sp.get_seasonal_holdout_split(df, horizon=horizon)
        train_df = train_split[train_split[valid_col] == 1].copy()
        cal_df   = cal_split[cal_split[valid_col] == 1].copy()
        test_df  = test_split[test_split[valid_col] == 1].copy()
        test_df  = test_df.sort_values("Date").reset_index(drop=True)

        X_tr  = train_df[feature_cols].to_numpy(dtype=np.float64)
        y_tr  = train_df[target_col].to_numpy(dtype=np.float64)
        X_cal = cal_df[feature_cols].to_numpy(dtype=np.float64)
        y_cal = cal_df[target_col].to_numpy(dtype=np.float64)
        X_ts  = test_df[feature_cols].to_numpy(dtype=np.float64)
        y_ts  = test_df[target_col].to_numpy(dtype=np.float64)

        cities = sorted(test_df["City"].unique())

        ridge = ModelFactory.create_model("Ridge", random_state=42)
        ridge.fit(X_tr, y_tr)
        ridge_preds_cal = ridge.predict(X_cal)
        ridge_preds_ts  = ridge.predict(X_ts)

        gb = ModelFactory.create_model("GradientBoosting", random_state=42)
        gb.fit(X_tr, y_tr)
        gb_preds_cal = gb.predict(X_cal)
        gb_preds_ts  = gb.predict(X_ts)

        for model_name, preds_cal, preds_ts in [
            ("Ridge", ridge_preds_cal, ridge_preds_ts),
            ("GradientBoosting", gb_preds_cal, gb_preds_ts),
        ]:
            cal_residuals = np.abs(y_cal - preds_cal)
            for w in windows:
                roll_conf = RollingConformalPredictor(
                    alpha=ALPHA, window_size=w, horizon=horizon, mode="city_specific"
                )
                lo_roll, up_roll, _ = roll_conf.run_rolling_evaluation(
                    test_df=test_df,
                    predictions=preds_ts,
                    y_true=y_ts,
                    initial_cal_scores=cal_residuals,
                    initial_cal_cities=cal_df["City"].to_numpy()
                )
                int_pool = calculate_interval_metrics(y_ts, lo_roll, up_roll, nominal=NOMINAL)
                records.append({
                    "Horizon": f"H{horizon}",
                    "Model": model_name,
                    "Window": w,
                    "City": "Pooled",
                    "PICP": int_pool["PICP"],
                    "MPIW": int_pool["MPIW"],
                })
                for c in cities:
                    c_mask = (test_df["City"] == c).to_numpy()
                    int_c = calculate_interval_metrics(y_ts[c_mask], lo_roll[c_mask], up_roll[c_mask], nominal=NOMINAL)
                    records.append({
                        "Horizon": f"H{horizon}",
                        "Model": model_name,
                        "Window": w,
                        "City": c,
                        "PICP": int_c["PICP"],
                        "MPIW": int_c["MPIW"],
                    })

    ablation_df = pd.DataFrame(records)
    out_path = os.path.join(results_dir, "ablation_conformal_window_held_out_winter.csv")
    ablation_df.to_csv(out_path, index=False)
    print(f"\nSaved held-out winter window ablation to {out_path}", flush=True)
    return ablation_df


if __name__ == "__main__":
    run_seasonal_holdout_per_city()
    run_seasonal_holdout_window_ablation()
