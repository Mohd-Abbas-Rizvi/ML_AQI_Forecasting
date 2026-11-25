# Fixed Split Conformal Prediction and Calendar-Accurate Rolling Adaptive Conformal Recalibration
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


class SplitConformalPredictor:
    # Fixed Split Conformal Prediction using held-out calibration set
    def __init__(self, alpha: float = 0.10):
        self.alpha = float(alpha)
        self.nominal_coverage = 1.0 - self.alpha
        self.q_hat: Optional[float] = None
        self.cal_scores = np.array([])

    def calibrate(self, y_cal: np.ndarray, y_pred_cal: np.ndarray):
        y_cal = np.asarray(y_cal, dtype=float)
        y_pred_cal = np.asarray(y_pred_cal, dtype=float)

        mask = ~np.isnan(y_cal) & ~np.isnan(y_pred_cal)
        scores = np.abs(y_cal[mask] - y_pred_cal[mask])
        self.cal_scores = np.sort(scores)

        n = len(self.cal_scores)
        if n == 0:
            raise ValueError("Conformal calibration failed: Calibration set contains no valid, non-NaN observations.")

        # Finite-sample conformal quantile: ceil((n + 1)*(1 - alpha)) / n
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(self.cal_scores, q_level, method="higher"))
        return self

    def predict_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.q_hat is None:
            raise RuntimeError("SplitConformalPredictor must be calibrated with calibrate() before predicting intervals.")
        y_pred = np.asarray(y_pred, dtype=float)
        lower = np.maximum(0.0, y_pred - self.q_hat)
        upper = y_pred + self.q_hat
        return lower, upper


class RollingConformalPredictor:
    # Adaptive Rolling Conformal Recalibration respecting calendar dates and multi-city grouping
    # Feedback arrival: A forecast made at origin date d for city c predicting target date d_target
    # (where d_target = d + horizon days) has its residual revealed ONLY when calendar date D >= d_target.
    def __init__(
        self,
        alpha: float = 0.10,
        window_size: int = 120,
        horizon: int = 1,
        mode: str = "pooled",  # "pooled" (cross-city buffer) or "city_specific"
    ):
        self.alpha = float(alpha)
        self.window_size = window_size
        self.horizon = horizon
        self.mode = mode
        self.nominal_coverage = 1.0 - self.alpha

    def run_rolling_evaluation(
        self,
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        y_true: np.ndarray,
        initial_cal_scores: np.ndarray,
        initial_cal_cities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Evaluates adaptive conformal intervals by stepping through chronological calendar dates
        if len(initial_cal_scores) == 0 or np.all(np.isnan(initial_cal_scores)):
            raise ValueError("Rolling conformal evaluation failed: Initial calibration scores array is empty.")

        clean_initial = np.asarray(initial_cal_scores)[~np.isnan(initial_cal_scores)]
        if len(clean_initial) == 0:
            raise ValueError("Rolling conformal evaluation failed: No valid non-NaN initial calibration scores.")

        # Ensure required columns
        df_eval = test_df.copy()
        df_eval["__orig_idx__"] = np.arange(len(df_eval))
        df_eval["__pred__"] = predictions
        df_eval["__y_true__"] = y_true

        if "Date" not in df_eval.columns:
            raise KeyError("test_df must contain a 'Date' column representing forecast origin.")
        df_eval["__origin_date__"] = pd.to_datetime(df_eval["Date"])

        target_date_col = f"target_H{self.horizon}_date"
        if target_date_col in df_eval.columns:
            df_eval["__target_date__"] = pd.to_datetime(df_eval[target_date_col])
        else:
            df_eval["__target_date__"] = df_eval["__origin_date__"] + pd.Timedelta(days=self.horizon)

        if "City" not in df_eval.columns:
            df_eval["City"] = "DefaultCity"

        # Initialize rolling buffers
        # If city_specific, maintain a buffer per city; if pooled, maintain a single global buffer
        if self.mode == "city_specific":
            buffers: Dict[str, List[float]] = {}
            for city in df_eval["City"].unique():
                if initial_cal_cities is not None:
                    c_mask = (initial_cal_cities == city) & (~np.isnan(initial_cal_scores))
                    c_scores = list(initial_cal_scores[c_mask])
                    buffers[city] = c_scores[-self.window_size:] if len(c_scores) > 0 else list(clean_initial[-self.window_size:])
                else:
                    buffers[city] = list(clean_initial[-self.window_size:])
        else:
            # Pooled buffer
            pooled_buffer: List[float] = list(clean_initial[-self.window_size:])

        # Pending forecasts queue: list of dicts:
        # {city, target_date, abs_residual}
        pending_forecasts: List[Dict] = []

        # Output arrays aligned with original test_df ordering
        n = len(df_eval)
        lower_bounds = np.zeros(n)
        upper_bounds = np.zeros(n)
        q_history = np.zeros(n)

        # Unique forecast origin dates in chronological order
        unique_dates = np.sort(df_eval["__origin_date__"].unique())

        for curr_date in unique_dates:
            curr_date = pd.to_datetime(curr_date)

            # Step 1: Reveal completed targets where target_date <= curr_date
            revealed = []
            remaining = []
            for item in pending_forecasts:
                if item["target_date"] <= curr_date:
                    revealed.append(item)
                else:
                    remaining.append(item)
            pending_forecasts = remaining

            # Update buffer with revealed residuals
            for item in revealed:
                res = item["residual"]
                if self.mode == "city_specific":
                    c = item["city"]
                    if c in buffers:
                        buffers[c].append(res)
                        if len(buffers[c]) > self.window_size:
                            buffers[c].pop(0)
                else:
                    pooled_buffer.append(res)
                    if len(pooled_buffer) > self.window_size:
                        pooled_buffer.pop(0)

            # Step 2: For all rows originating on curr_date, compute conformal intervals
            day_rows = df_eval[df_eval["__origin_date__"] == curr_date]

            for _, row in day_rows.iterrows():
                orig_idx = int(row["__orig_idx__"])
                city = str(row["City"])
                y_hat = float(row["__pred__"])
                y_act = float(row["__y_true__"])
                target_dt = pd.to_datetime(row["__target_date__"])

                # Get active buffer
                if self.mode == "city_specific":
                    active_b = buffers.get(city, list(clean_initial))
                else:
                    active_b = pooled_buffer

                if len(active_b) == 0:
                    raise ValueError(f"Rolling conformal active buffer is empty on date {curr_date} for city {city}.")

                # Compute finite-sample quantile from active buffer
                b_arr = np.sort(np.asarray(active_b))
                b_len = len(b_arr)
                q_level = min(1.0, np.ceil((b_len + 1) * (1.0 - self.alpha)) / b_len)
                q_t = float(np.quantile(b_arr, q_level, method="higher"))

                lower_bounds[orig_idx] = max(0.0, y_hat - q_t)
                upper_bounds[orig_idx] = y_hat + q_t
                q_history[orig_idx] = q_t

                # Step 3: Enqueue this forecast's residual to be revealed at target_date (if target is observed)
                if not np.isnan(y_act) and not np.isnan(y_hat):
                    abs_res = abs(y_act - y_hat)
                    pending_forecasts.append({
                        "city": city,
                        "target_date": target_dt,
                        "residual": abs_res,
                    })

        return lower_bounds, upper_bounds, q_history
