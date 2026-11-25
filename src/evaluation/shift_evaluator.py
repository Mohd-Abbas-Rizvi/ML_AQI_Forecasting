# Automated Distribution-Shift Evaluator and Benchmark Engine
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import torch
torch.set_num_threads(1)

from src.models.baselines import PersistenceForecaster, SeasonalNaiveForecaster
from src.models.scratch_numpy import NumPyRidgeRegression, NumPyMLPRegressor
from src.models.classical import ModelFactory, get_feature_columns
from src.models.neural import PyTorchMLPWrapper
from src.uncertainty.conformal import SplitConformalPredictor, RollingConformalPredictor
from src.uncertainty.quantile import QuantileRegressorWrapper
from src.evaluation.metrics import (
    calculate_point_metrics,
    calculate_interval_metrics,
    calculate_severity_metrics,
    calculate_shift_penalty,
    calculate_pinball_loss,
)
from src.data.splitters import TemporalSplitter, ShiftScenarioPartitioner


class ShiftEvaluationEngine:
    def __init__(
        self,
        features_path: str = "data/processed/features_ffill_train_median.csv",
        models_dir: str = "models",
        results_dir: str = "results",
        seed: int = 42,
    ):
        self.features_path = features_path
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.seed = seed
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

        self.df = pd.read_csv(features_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.feature_cols = get_feature_columns(self.df)
        self.splitter = TemporalSplitter()
        print(f"Loaded feature dataset: {len(self.df)} rows, {len(self.feature_cols)} predictive features.")

    def run_all_models_for_horizon(self, horizon: int = 1) -> pd.DataFrame:
        target_col = f"target_H{horizon}"
        valid_col = f"target_H{horizon}_valid"
        print(f"\n=======================================================", flush=True)
        print(f"  RUNNING COMPLETE EVALUATION FOR HORIZON H={horizon}", flush=True)
        print(f"=======================================================", flush=True)

        # 1. Obtain strictly verified chronological partitions for this horizon
        train_split, val_split, test_split = self.splitter.get_splits(self.df, horizon=horizon)
        print(f"Chronological Splits (H={horizon}): Proper Train={len(train_split)}, Val/Cal={len(val_split)}, Test={len(test_split)}", flush=True)

        tr_valid = train_split[train_split[valid_col] == 1].copy()
        val_valid = val_split[val_split[valid_col] == 1].copy()
        ts_valid = test_split[test_split[valid_col] == 1].copy()

        X_tr = tr_valid[self.feature_cols].to_numpy(dtype=np.float64)
        y_tr = tr_valid[target_col].to_numpy(dtype=np.float64)

        X_val = val_valid[self.feature_cols].to_numpy(dtype=np.float64)
        y_val = val_valid[target_col].to_numpy(dtype=np.float64)

        X_ts = ts_valid[self.feature_cols].to_numpy(dtype=np.float64)
        y_ts = ts_valid[target_col].to_numpy(dtype=np.float64)

        # 2. Instantiate and train all primary chronological models
        models = {
            "Persistence": PersistenceForecaster(),
            "SeasonalNaive": SeasonalNaiveForecaster(),
            "NumPy_Ridge": NumPyRidgeRegression(alpha=10.0),
            "NumPy_MLP": NumPyMLPRegressor(hidden_dim=64, lr=0.005, epochs=100, random_state=self.seed),
            "Ridge": ModelFactory.create_model("Ridge", random_state=self.seed),
            "RandomForest": ModelFactory.create_model("RandomForest", random_state=self.seed),
            "GradientBoosting": ModelFactory.create_model("GradientBoosting", random_state=self.seed),
            "XGBoost": ModelFactory.create_model("XGBoost", random_state=self.seed),
            "PyTorch_MLP": PyTorchMLPWrapper(hidden_dim=64, epochs=30, random_state=self.seed),
            "Quantile_GBM": QuantileRegressorWrapper(random_state=self.seed),
        }

        val_preds_dict = {}
        test_preds_dict = {}

        for mname, model in models.items():
            print(f"--> Training {mname} (H={horizon})...", flush=True)
            if mname in ["Persistence", "SeasonalNaive"]:
                val_pred = model.predict(val_valid)
                test_pred = model.predict(ts_valid)
            elif mname in ["NumPy_Ridge", "NumPy_MLP", "PyTorch_MLP"]:
                model.fit(X_tr, y_tr)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_ts)
            elif mname == "Quantile_GBM":
                model.fit(X_tr, y_tr)
                _, val_pred, _ = model.predict(X_val)
                _, test_pred, _ = model.predict(X_ts)
            else:
                model.fit(X_tr, y_tr)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_ts)

            val_preds_dict[mname] = val_pred
            test_preds_dict[mname] = test_pred

            save_path = os.path.join(self.models_dir, f"{mname}_H{horizon}.joblib")
            joblib.dump(model, save_path)
            print(f"  [OK] {mname} fitted and saved to {save_path}", flush=True)

        # 3. Setup Conformal Predictors using held-out validation/calibration split
        conformal_predictors = {}
        rolling_intervals = {}

        for mname in models.keys():
            val_p = val_preds_dict[mname]
            sp = SplitConformalPredictor(alpha=0.10)
            sp.calibrate(y_val, val_p)
            conformal_predictors[mname] = sp

            # Precompute rolling conformal intervals on the full chronological test set using calendar dates
            val_scores = np.abs(y_val - val_p)
            val_cities = val_valid["City"].to_numpy()
            rcp = RollingConformalPredictor(alpha=0.10, window_size=120, horizon=horizon, mode="pooled")
            lo_full, up_full, _ = rcp.run_rolling_evaluation(
                ts_valid, test_preds_dict[mname], y_ts, val_scores, val_cities
            )
            rolling_intervals[mname] = (lo_full, up_full)

        # 4. Construct Independently Fitted Random-Split Reference Benchmark
        # Ensures zero training leakage into the random reference test set
        print("--> Fitting independent Random Split reference benchmark...", flush=True)
        all_valid_df = self.df[self.df[valid_col] == 1].copy()
        np.random.seed(self.seed)
        shuffled_indices = np.random.permutation(len(all_valid_df))
        n_rand_test = len(ts_valid)
        rand_test_idx = shuffled_indices[:n_rand_test]
        rand_train_idx = shuffled_indices[n_rand_test:]

        rand_train_df = all_valid_df.iloc[rand_train_idx].copy()
        rand_test_df = all_valid_df.iloc[rand_test_idx].copy()

        X_rand_tr = rand_train_df[self.feature_cols].to_numpy(dtype=np.float64)
        y_rand_tr = rand_train_df[target_col].to_numpy(dtype=np.float64)
        X_rand_ts = rand_test_df[self.feature_cols].to_numpy(dtype=np.float64)
        y_rand_ts = rand_test_df[target_col].to_numpy(dtype=np.float64)

        # Fit separate reference models strictly on rand_train_df
        ref_maes = {}
        ref_models = {
            "Persistence": PersistenceForecaster(),
            "SeasonalNaive": SeasonalNaiveForecaster(),
            "NumPy_Ridge": NumPyRidgeRegression(alpha=10.0),
            "NumPy_MLP": NumPyMLPRegressor(hidden_dim=64, lr=0.005, epochs=100, random_state=self.seed),
            "Ridge": ModelFactory.create_model("Ridge", random_state=self.seed),
            "RandomForest": ModelFactory.create_model("RandomForest", random_state=self.seed),
            "GradientBoosting": ModelFactory.create_model("GradientBoosting", random_state=self.seed),
            "XGBoost": ModelFactory.create_model("XGBoost", random_state=self.seed),
            "PyTorch_MLP": PyTorchMLPWrapper(hidden_dim=64, epochs=30, random_state=self.seed),
            "Quantile_GBM": QuantileRegressorWrapper(random_state=self.seed),
        }

        for mname, r_model in ref_models.items():
            if mname in ["Persistence", "SeasonalNaive"]:
                p_rand = r_model.predict(rand_test_df)
            elif mname in ["NumPy_Ridge", "NumPy_MLP", "PyTorch_MLP"]:
                r_model.fit(X_rand_tr, y_rand_tr)
                p_rand = r_model.predict(X_rand_ts)
            elif mname == "Quantile_GBM":
                r_model.fit(X_rand_tr, y_rand_tr)
                _, p_rand, _ = r_model.predict(X_rand_ts)
            else:
                r_model.fit(X_rand_tr, y_rand_tr)
                p_rand = r_model.predict(X_rand_ts)
            ref_maes[mname] = calculate_point_metrics(y_rand_ts, p_rand)["MAE"]

        # 5. Construct Shift Slices
        slices = {
            "Random_Reference": (rand_test_df, X_rand_ts, y_rand_ts, True),
            "Chronological_Full": (ts_valid, X_ts, y_ts, False),
        }

        for sname, sdf in ShiftScenarioPartitioner.get_year_slices(ts_valid).items():
            slices[sname] = (sdf, sdf[self.feature_cols].to_numpy(dtype=np.float64), sdf[target_col].to_numpy(dtype=np.float64), False)

        for sname, sdf in ShiftScenarioPartitioner.get_seasonal_slices(ts_valid).items():
            slices[sname] = (sdf, sdf[self.feature_cols].to_numpy(dtype=np.float64), sdf[target_col].to_numpy(dtype=np.float64), False)

        for sname, sdf in ShiftScenarioPartitioner.get_event_slices(ts_valid).items():
            slices[sname] = (sdf, sdf[self.feature_cols].to_numpy(dtype=np.float64), sdf[target_col].to_numpy(dtype=np.float64), False)

        for sname, sdf in ShiftScenarioPartitioner.get_city_slices(ts_valid).items():
            slices[sname] = (sdf, sdf[self.feature_cols].to_numpy(dtype=np.float64), sdf[target_col].to_numpy(dtype=np.float64), False)

        print(f"--> Slices constructed: {list(slices.keys())}", flush=True)

        # 6. Evaluate all models across all slices
        results_records = []

        for sname, (slice_df, X_s, y_s, is_rand) in slices.items():
            for mname, model in models.items():
                if is_rand:
                    r_model = ref_models[mname]
                    if mname in ["Persistence", "SeasonalNaive"]:
                        preds = r_model.predict(slice_df)
                    elif mname == "Quantile_GBM":
                        _, preds, _ = r_model.predict(X_s)
                    else:
                        preds = r_model.predict(X_s)
                else:
                    sub_idx = ts_valid.index.get_indexer(slice_df.index)
                    preds = test_preds_dict[mname][sub_idx]

                pt = calculate_point_metrics(y_s, preds)
                shift_pen = calculate_shift_penalty(pt["MAE"], ref_maes[mname])

                # Static Conformal
                sp = conformal_predictors[mname]
                lo_static, up_static = sp.predict_intervals(preds)
                int_static = calculate_interval_metrics(y_s, lo_static, up_static, nominal=0.90)

                # Rolling Conformal (calendar-accurate slice lookup)
                if not is_rand:
                    lo_full, up_full = rolling_intervals[mname]
                    lo_roll = lo_full[sub_idx]
                    up_roll = up_full[sub_idx]
                    int_roll = calculate_interval_metrics(y_s, lo_roll, up_roll, nominal=0.90)
                else:
                    int_roll = int_static

                sev = calculate_severity_metrics(y_s, preds, severe_threshold=400.0)

                rec = {
                    "Horizon": f"H{horizon}",
                    "Scenario": sname,
                    "Model": mname,
                    "MAE": pt["MAE"],
                    "RMSE": pt["RMSE"],
                    "R2": pt["R2"],
                    "ShiftPenalty": shift_pen,
                    "SampleCount": pt["Count"],
                    "PICP_Static": int_static["PICP"],
                    "MPIW_Static": int_static["MPIW"],
                    "CoverageGap_Static": int_static["CoverageGap"],
                    "PICP_Rolling": int_roll["PICP"],
                    "MPIW_Rolling": int_roll["MPIW"],
                    "CoverageGap_Rolling": int_roll["CoverageGap"],
                    "Severe_Recall": sev.get("Severe_Recall", np.nan),
                    "Severe_Precision": sev.get("Severe_Precision", np.nan),
                    "Severe_F1": sev.get("Severe_F1", np.nan),
                    "MAE_Severe": sev.get("MAE_Severe", np.nan),
                    "MAE_VeryPoor": sev.get("MAE_VeryPoor", np.nan),
                    "MAE_Poor": sev.get("MAE_Poor", np.nan),
                    "MAE_Moderate": sev.get("MAE_Moderate", np.nan),
                }

                if mname == "Quantile_GBM":
                    if is_rand:
                        lo_q, med_q, up_q = ref_models["Quantile_GBM"].predict(X_s)
                    else:
                        lo_q, med_q, up_q = model.predict(X_s)
                    rec["Pinball_0.05"] = calculate_pinball_loss(y_s, lo_q, 0.05)
                    rec["Pinball_0.50"] = calculate_pinball_loss(y_s, med_q, 0.50)
                    rec["Pinball_0.95"] = calculate_pinball_loss(y_s, up_q, 0.95)

                results_records.append(rec)

        res_df = pd.DataFrame(results_records)
        save_table = os.path.join(self.results_dir, "tables", f"metrics_H{horizon}.csv")
        res_df.to_csv(save_table, index=False)
        print(f"Saved metrics for Horizon H={horizon} to {save_table}", flush=True)
        return res_df


def run_full_shift_experiments():
    engine = ShiftEvaluationEngine()
    df_h1 = engine.run_all_models_for_horizon(horizon=1)
    df_h3 = engine.run_all_models_for_horizon(horizon=3)
    combined = pd.concat([df_h1, df_h3], ignore_index=True)
    combined_path = os.path.join("results", "tables", "metrics_all_horizons.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nSuccessfully finished all baseline, shift, and conformal experiments!", flush=True)
    print(f"Saved master results to {combined_path}", flush=True)
    return combined


if __name__ == "__main__":
    run_full_shift_experiments()
