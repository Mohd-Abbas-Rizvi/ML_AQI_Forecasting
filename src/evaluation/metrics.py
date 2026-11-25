# Performance, Uncertainty, and Severity-aware Evaluation Metrics
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List


def calculate_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Calculate MAE, RMSE, and R2
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Count": 0}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    ss_res = np.sum((yt - yp) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Count": int(len(yt))}


def calculate_interval_metrics(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float = 0.90
) -> Dict[str, float]:
    # Calculate PICP, MPIW, and Coverage Gap
    mask = ~np.isnan(y_true) & ~np.isnan(lower) & ~np.isnan(upper)
    if not np.any(mask):
        return {"PICP": np.nan, "MPIW": np.nan, "CoverageGap": np.nan, "Count": 0}
    yt = y_true[mask]
    lo = lower[mask]
    up = upper[mask]
    covered = (yt >= lo) & (yt <= up)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(up - lo))
    gap = float(picp - nominal)
    return {"PICP": picp, "MPIW": mpiw, "CoverageGap": gap, "Count": int(len(yt))}


def calculate_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    # Pinball / quantile loss for quantile evaluation
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return np.nan
    diff = y_true[mask] - y_pred[mask]
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))


def calculate_severity_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, severe_threshold: float = 400.0
) -> Dict[str, float]:
    # Compute per-bucket MAE and severe day alert detection metrics
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return {}
    yt = y_true[mask]
    yp = y_pred[mask]

    # AQI Buckets
    buckets = {
        "Good_Sat": (0.0, 100.0),
        "Moderate": (101.0, 200.0),
        "Poor": (201.0, 300.0),
        "VeryPoor": (301.0, 400.0),
        "Severe": (401.0, 10000.0),
    }

    metrics = {}
    for bname, (lo, hi) in buckets.items():
        bmask = (yt >= lo) & (yt <= hi)
        if np.any(bmask):
            metrics[f"MAE_{bname}"] = float(np.mean(np.abs(yt[bmask] - yp[bmask])))
            metrics[f"Count_{bname}"] = int(np.sum(bmask))
        else:
            metrics[f"MAE_{bname}"] = np.nan
            metrics[f"Count_{bname}"] = 0

    # Severe alert detection (True class: yt > severe_threshold, Predicted alert: yp > severe_threshold)
    actual_severe = yt > severe_threshold
    pred_severe = yp > severe_threshold

    tp = np.sum(actual_severe & pred_severe)
    fp = np.sum(~actual_severe & pred_severe)
    fn = np.sum(actual_severe & ~pred_severe)
    tn = np.sum(~actual_severe & ~pred_severe)

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics["Severe_TP"] = int(tp)
    metrics["Severe_FP"] = int(fp)
    metrics["Severe_FN"] = int(fn)
    metrics["Severe_Recall"] = recall
    metrics["Severe_Precision"] = precision
    metrics["Severe_F1"] = f1

    return metrics


def calculate_shift_penalty(mae_shifted: float, mae_reference: float) -> float:
    # Compute shift penalty: MAE under shift / MAE under reference evaluation
    if mae_reference <= 0.0 or np.isnan(mae_reference) or np.isnan(mae_shifted):
        return np.nan
    return float(mae_shifted / mae_reference)
