"""
Fix 2 & 3: Temporal Stability Analysis & Validation-Selected Block-Bootstrap
=============================================================================
Fix 2 Requirements:
- Sort test rows by DATE.
- Build 5 contiguous TIME blocks across the 548 calendar test days.
- Deterministic models (Persistence, SeasonalNaive, NumPy_Ridge, Ridge, GradientBoosting):
    Evaluated across the 5 contiguous time blocks -> MAE (mean +/- std), ShiftPenalty (mean +/- std).
- Stochastic models (RandomForest, XGBoost, PyTorch_MLP):
    Evaluated across 5 random seeds [42, 123, 456, 789, 1024] -> MAE (mean +/- std), ShiftPenalty (mean +/- std).

Fix 3 Requirements:
- Paired block bootstrap per city, sorted by date.
- Model selection: SELECT BEST MODEL PER CITY ON VALIDATION SET (NOT TEST!).
- Windows evaluated per city:
    1. Chronological_Full (2019-2020)
    2. HeldOut_Winter (Nov-Feb 2019-2020)
    3. Event_Diwali
    4. Event_StubbleBurning
    5. Event_Lockdown
    6. Event_WinterInversion
- 95% Confidence Intervals for MAE difference vs Persistence (Diff_MAE = MAE_best - MAE_pers)
  for both H1 and H3.

Saves:
  results/tables/stability_analysis.csv
  results/tables/bootstrap_ci_per_city.csv
  results/tables/bootstrap_ci.csv
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
from src.models.baselines import PersistenceForecaster, SeasonalNaiveForecaster
from src.models.scratch_numpy import NumPyRidgeRegression
from src.models.classical import ModelFactory, get_feature_columns
from src.models.neural import PyTorchMLPWrapper
from src.evaluation.metrics import calculate_point_metrics, calculate_shift_penalty

SEEDS = [42, 123, 456, 789, 1024]
N_BOOTSTRAP = 1000
N_TIME_BLOCKS = 5


def moving_block_bootstrap_diff(e_model, e_pers, B=N_BOOTSTRAP, block_size=14, alpha=0.05):
    """
    Moving block bootstrap for paired error difference: e_model - e_pers.
    Returns:
      obs_diff: mean(e_model) - mean(e_pers)
      ci_lo, ci_hi: 95% CI of the difference
      obs_pct: (mean(e_pers) - mean(e_model)) / mean(e_pers) * 100
      pct_lo, pct_hi: 95% CI of percentage improvement
    """
    n = len(e_model)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    m_m = float(np.mean(e_model))
    m_p = float(np.mean(e_pers))
    obs_diff = m_m - m_p
    obs_pct = ((m_p - m_m) / m_p * 100.0) if m_p > 0 else 0.0

    if n < 5:
        return obs_diff, obs_diff, obs_diff, obs_pct, obs_pct, obs_pct

    b_size = max(1, min(block_size, n // 3))
    n_blocks = int(np.ceil(n / b_size))
    rng = np.random.default_rng(42)

    diffs = np.empty(B)
    pcts  = np.empty(B)

    # Circular array for boundary sampling
    e_m_circ = np.concatenate([e_model, e_model[:b_size]])
    e_p_circ = np.concatenate([e_pers,  e_pers[:b_size]])

    for b in range(B):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + b_size) for s in starts])[:n]

        sample_m = np.mean(e_m_circ[idx])
        sample_p = np.mean(e_p_circ[idx])
        diffs[b] = sample_m - sample_p
        pcts[b]  = ((sample_p - sample_m) / sample_p * 100.0) if sample_p > 0 else 0.0

    ci_lo = float(np.percentile(diffs, 100 * alpha / 2))
    ci_hi = float(np.percentile(diffs, 100 * (1.0 - alpha / 2)))
    pct_lo = float(np.percentile(pcts, 100 * alpha / 2))
    pct_hi = float(np.percentile(pcts, 100 * (1.0 - alpha / 2)))

    return obs_diff, ci_lo, ci_hi, obs_pct, pct_lo, pct_hi


def run_complete_stability_and_bootstrap(
    features_path="data/processed/features_ffill_train_median.csv",
    results_dir="results/tables"
):
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    feature_cols = get_feature_columns(df)
    sp = TemporalSplitter()

    # Pre-load verified reference MAEs from metrics_all_horizons.csv
    metrics_path = os.path.join(results_dir, "metrics_all_horizons.csv")
    ref_maes_dict = {}
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        ref_df = m_df[m_df["Scenario"] == "Random_Reference"]
        for _, row in ref_df.iterrows():
            ref_maes_dict[(row["Horizon"], row["Model"])] = float(row["MAE"])

    stability_rows = []
    city_bootstrap_rows = []
    pooled_bootstrap_rows = []

    for horizon in [1, 3]:
        print(f"\n=======================================================", flush=True)
        print(f"  HORIZON H={horizon}: STABILITY & BOOTSTRAP ANALYSIS", flush=True)
        print(f"=======================================================", flush=True)

        target_col = f"target_H{horizon}"
        valid_col  = f"target_H{horizon}_valid"

        train_split, val_split, test_split = sp.get_splits(df, horizon=horizon)

        tr_valid  = train_split[train_split[valid_col] == 1].copy()
        val_valid = val_split[val_split[valid_col] == 1].copy()
        ts_valid  = test_split[test_split[valid_col] == 1].copy()

        # SORT TEST ROWS BY DATE to guarantee non-city-ordered blocks
        ts_valid = ts_valid.sort_values(["Date", "City"]).reset_index(drop=True)

        X_tr = tr_valid[feature_cols].to_numpy(dtype=np.float64)
        y_tr = tr_valid[target_col].to_numpy(dtype=np.float64)
        X_val = val_valid[feature_cols].to_numpy(dtype=np.float64)
        y_val = val_valid[target_col].to_numpy(dtype=np.float64)
        X_ts = ts_valid[feature_cols].to_numpy(dtype=np.float64)
        y_ts = ts_valid[target_col].to_numpy(dtype=np.float64)

        # Persistence on test
        pers = PersistenceForecaster()
        pers.fit(tr_valid, tr_valid[target_col])
        pers_preds_ts = pers.predict(ts_valid)
        pers_mae_ts   = float(np.mean(np.abs(y_ts - pers_preds_ts)))
        pers_ref_mae  = ref_maes_dict.get((f"H{horizon}", "Persistence"), 27.91 if horizon == 1 else 41.38)

        # ---------------------------------------------------------------------
        # PART 1 (Fix 2): Deterministic Models on 5 Contiguous TIME Blocks
        # ---------------------------------------------------------------------
        unique_dates = np.sort(ts_valid["Date"].unique())
        date_blocks  = np.array_split(unique_dates, N_TIME_BLOCKS)
        block_labels = [f"{pd.to_datetime(b[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(b[-1]).strftime('%Y-%m-%d')}" for b in date_blocks]
        print(f"  Constructed 5 Contiguous Date Blocks: {block_labels}", flush=True)

        # Fit all primary candidate models once
        fitted_models = {
            "Persistence": pers,
            "SeasonalNaive": SeasonalNaiveForecaster().fit(tr_valid, tr_valid[target_col]),
            "NumPy_Ridge": NumPyRidgeRegression(alpha=10.0).fit(X_tr, y_tr),
            "Ridge": ModelFactory.create_model("Ridge", random_state=42).fit(X_tr, y_tr),
            "GradientBoosting": ModelFactory.create_model("GradientBoosting", random_state=42).fit(X_tr, y_tr),
            "RandomForest": ModelFactory.create_model("RandomForest", random_state=42).fit(X_tr, y_tr),
            "XGBoost": ModelFactory.create_model("XGBoost", random_state=42).fit(X_tr, y_tr),
            "PyTorch_MLP": PyTorchMLPWrapper(hidden_dim=64, epochs=15, random_state=42).fit(X_tr, y_tr),
        }

        # Deterministic models evaluate across 5 TIME BLOCKS
        deterministic_models = ["Persistence", "SeasonalNaive", "NumPy_Ridge", "Ridge", "GradientBoosting"]
        for mname in deterministic_models:
            m = fitted_models[mname]
            if mname in ["Persistence", "SeasonalNaive"]:
                preds = m.predict(ts_valid)
            else:
                preds = m.predict(X_ts)

            block_maes = []
            block_sps  = []
            r_mae = ref_maes_dict.get((f"H{horizon}", mname), pers_ref_mae)

            for b_dates in date_blocks:
                b_mask = ts_valid["Date"].isin(b_dates).to_numpy()
                b_y = y_ts[b_mask]
                b_p = preds[b_mask]
                b_mae = float(np.mean(np.abs(b_y - b_p)))
                block_maes.append(b_mae)
                block_sps.append(calculate_shift_penalty(b_mae, r_mae))

            overall_mae = float(np.mean(np.abs(y_ts - preds)))
            overall_sp  = calculate_shift_penalty(overall_mae, r_mae)

            stability_rows.append({
                "Horizon": f"H{horizon}",
                "Model": mname,
                "EvaluationMode": "5_Contiguous_Time_Blocks",
                "MAE_Overall": overall_mae,
                "MAE_Mean": float(np.mean(block_maes)),
                "MAE_Std": float(np.std(block_maes, ddof=1)),
                "ShiftPenalty_Overall": overall_sp,
                "ShiftPenalty_Mean": float(np.mean(block_sps)),
                "ShiftPenalty_Std": float(np.std(block_sps, ddof=1)),
            })

        # ---------------------------------------------------------------------
        # PART 2 (Fix 2): Stochastic Models on 5 Random Seeds
        # ---------------------------------------------------------------------
        stochastic_model_factories = {
            "RandomForest": lambda seed: ModelFactory.create_model("RandomForest", random_state=seed),
            "XGBoost": lambda seed: ModelFactory.create_model("XGBoost", random_state=seed),
            "PyTorch_MLP": lambda seed: PyTorchMLPWrapper(hidden_dim=64, epochs=15, random_state=seed),
        }

        for mname, factory in stochastic_model_factories.items():
            print(f"  Evaluating stochastic model {mname} across 5 seeds...", flush=True)
            seed_maes = []
            seed_sps  = []
            r_mae = ref_maes_dict.get((f"H{horizon}", mname), pers_ref_mae)

            for seed in SEEDS:
                m_seed = factory(seed)
                m_seed.fit(X_tr, y_tr)
                p_seed = m_seed.predict(X_ts)
                mae_s  = float(np.mean(np.abs(y_ts - p_seed)))
                seed_maes.append(mae_s)
                seed_sps.append(calculate_shift_penalty(mae_s, r_mae))

            overall_mae = seed_maes[0]
            overall_sp  = seed_sps[0]

            stability_rows.append({
                "Horizon": f"H{horizon}",
                "Model": mname,
                "EvaluationMode": "5_Random_Seeds",
                "MAE_Overall": overall_mae,
                "MAE_Mean": float(np.mean(seed_maes)),
                "MAE_Std": float(np.std(seed_maes, ddof=1)),
                "ShiftPenalty_Overall": overall_sp,
                "ShiftPenalty_Mean": float(np.mean(seed_sps)),
                "ShiftPenalty_Std": float(np.std(seed_sps, ddof=1)),
            })

        # ---------------------------------------------------------------------
        # PART 3 (Fix 3): Model Selection on Validation per City
        # ---------------------------------------------------------------------
        cities = sorted(ts_valid["City"].unique())
        best_model_per_city = {}

        candidate_model_names = ["Ridge", "RandomForest", "GradientBoosting", "XGBoost", "PyTorch_MLP"]
        print(f"\n  Model Selection on Validation Set for H={horizon}:", flush=True)

        for city in cities:
            c_val_mask = (val_valid["City"] == city).to_numpy()
            y_c_val = y_val[c_val_mask]
            X_c_val = X_val[c_val_mask]

            val_maes = {}
            for mname in candidate_model_names:
                m = fitted_models[mname]
                preds_val = m.predict(X_c_val)
                val_maes[mname] = float(np.mean(np.abs(y_c_val - preds_val)))

            best_m = min(val_maes, key=val_maes.get)
            best_model_per_city[city] = best_m
            print(f"    {city:10s} -> Best Model on Validation: {best_m:16s} (Val MAE={val_maes[best_m]:.2f})", flush=True)

        # ---------------------------------------------------------------------
        # PART 4 (Fix 3): Paired Block Bootstrap per City Across Evaluation Windows
        # ---------------------------------------------------------------------
        # Target dates for held-out winter filter
        ts_valid["__target_date__"] = pd.to_datetime(ts_valid["Date"]) + pd.Timedelta(days=horizon)

        for city in cities:
            best_mname = best_model_per_city[city]
            m_best = fitted_models[best_mname]

            # Predictions on all test rows for this city
            city_test = ts_valid[ts_valid["City"] == city].sort_values("Date").reset_index(drop=True)
            X_city_ts = city_test[feature_cols].to_numpy(dtype=np.float64)
            y_city_ts = city_test[target_col].to_numpy(dtype=np.float64)

            preds_best = m_best.predict(X_city_ts)
            preds_pers = pers.predict(city_test)

            e_best_all = np.abs(y_city_ts - preds_best)
            e_pers_all = np.abs(y_city_ts - preds_pers)

            # Define windows for this city
            windows = {
                "Chronological_Full": np.ones(len(city_test), dtype=bool),
                "HeldOut_Winter": city_test["__target_date__"].dt.month.isin([11, 12, 1, 2]) & city_test["__target_date__"].dt.year.isin([2019, 2020]),
                "Event_Diwali": city_test["is_diwali_window"] == 1,
                "Event_StubbleBurning": city_test["is_stubble_burning"] == 1,
                "Event_Lockdown": city_test["is_lockdown"] == 1,
                "Event_WinterInversion": city_test["is_winter_inversion"] == 1,
            }

            for win_name, win_mask in windows.items():
                win_mask_arr = win_mask.to_numpy() if isinstance(win_mask, pd.Series) else win_mask
                count = int(np.sum(win_mask_arr))
                if count < 5:
                    continue

                w_e_best = e_best_all[win_mask_arr]
                w_e_pers = e_pers_all[win_mask_arr]

                b_size = 14 if count >= 30 else (7 if count >= 15 else 3)
                diff_obs, ci_lo, ci_hi, pct_obs, pct_lo, pct_hi = moving_block_bootstrap_diff(
                    w_e_best, w_e_pers, B=N_BOOTSTRAP, block_size=b_size
                )

                city_bootstrap_rows.append({
                    "Horizon": f"H{horizon}",
                    "City": city,
                    "EvaluationWindow": win_name,
                    "SelectedModel": best_mname,
                    "SampleCount": count,
                    "MAE_SelectedModel": float(np.mean(w_e_best)),
                    "MAE_Persistence": float(np.mean(w_e_pers)),
                    "Diff_MAE": diff_obs,
                    "Diff_MAE_CI95_Lower": ci_lo,
                    "Diff_MAE_CI95_Upper": ci_hi,
                    "Pct_Improvement": pct_obs,
                    "Pct_Improvement_CI95_Lower": pct_lo,
                    "Pct_Improvement_CI95_Upper": pct_hi,
                    "Model_Beats_Persistence": "Yes" if diff_obs < 0 else "No",
                    "Statistically_Significant": "Yes" if (ci_hi < 0 or ci_lo > 0) else "No",
                })

        # Also compute pooled bootstrap for each primary model on Chronological_Full
        for mname in ["Ridge", "GradientBoosting", "XGBoost", "RandomForest", "PyTorch_MLP"]:
            m = fitted_models[mname]
            p = m.predict(X_ts)
            e_m = np.abs(y_ts - p)
            e_p = np.abs(y_ts - pers_preds_ts)

            diff_obs, ci_lo, ci_hi, pct_obs, pct_lo, pct_hi = moving_block_bootstrap_diff(
                e_m, e_p, B=N_BOOTSTRAP, block_size=14
            )

            meets_10 = "Yes (>= 10%)" if (pct_obs >= 10.0 and pct_lo >= 10.0) else ("Yes (Mean >= 10%)" if pct_obs >= 10.0 else "No")

            pooled_bootstrap_rows.append({
                "Horizon": f"H{horizon}",
                "Model": mname,
                "MAE_Model": float(np.mean(e_m)),
                "MAE_Persistence": pers_mae_ts,
                "Diff_MAE": diff_obs,
                "Diff_MAE_CI95_Lower": ci_lo,
                "Diff_MAE_CI95_Upper": ci_hi,
                "Pct_Improvement": pct_obs,
                "Pct_Improvement_CI95_Lower": pct_lo,
                "Pct_Improvement_CI95_Upper": pct_hi,
                "Meets_Criterion_10Pct": meets_10,
            })

    stab_df = pd.DataFrame(stability_rows)
    city_boot_df = pd.DataFrame(city_bootstrap_rows)
    pool_boot_df = pd.DataFrame(pooled_bootstrap_rows)

    stab_path = os.path.join(results_dir, "stability_analysis.csv")
    city_boot_path = os.path.join(results_dir, "bootstrap_ci_per_city.csv")
    pool_boot_path = os.path.join(results_dir, "bootstrap_ci.csv")

    stab_df.to_csv(stab_path, index=False)
    city_boot_df.to_csv(city_boot_path, index=False)
    pool_boot_df.to_csv(pool_boot_path, index=False)

    print(f"\nSaved updated stability analysis to {stab_path}", flush=True)
    print(stab_df.to_string(index=False), flush=True)

    print(f"\nSaved per-city bootstrap results to {city_boot_path}", flush=True)
    print(city_boot_df[city_boot_df["EvaluationWindow"].isin(["Chronological_Full", "HeldOut_Winter"])][
        ["Horizon", "City", "EvaluationWindow", "SelectedModel", "MAE_SelectedModel", "MAE_Persistence", "Diff_MAE", "Diff_MAE_CI95_Lower", "Diff_MAE_CI95_Upper", "Pct_Improvement"]
    ].to_string(index=False), flush=True)

    return stab_df, city_boot_df, pool_boot_df


if __name__ == "__main__":
    run_complete_stability_and_bootstrap()
