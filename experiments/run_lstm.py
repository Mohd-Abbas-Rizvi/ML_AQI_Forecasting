import os
import numpy as np
import pandas as pd

from src.models.lstm import LSTMForecasterWrapper
from src.data.splitters import TemporalSplitter, ShiftScenarioPartitioner
from src.uncertainty.conformal import (
    SplitConformalPredictor,
    RollingConformalPredictor,
)
from src.evaluation.metrics import (
    calculate_point_metrics,
    calculate_interval_metrics,
    calculate_severity_metrics,
    calculate_shift_penalty,
    calculate_pinball_loss,
)


FEATURES_PATH = "data/processed/features_ffill_train_median.csv"
RESULTS_PATH = "results/tables/lstm_metrics.csv"
MASTER_RESULTS_PATH = "results/tables/metrics_all_horizons.csv"
MODELS_DIR = "models"

SEED = 42


def evaluate_lstm_horizon(df, horizon):
    print()
    print("=" * 60)
    print(f"TRAINING AND EVALUATING LSTM FOR H={horizon}")
    print("=" * 60)

    target_col = f"target_H{horizon}"
    valid_col = f"target_H{horizon}_valid"

    splitter = TemporalSplitter()

    train_split, val_split, test_split = splitter.get_splits(
        df,
        horizon=horizon,
    )

    train_valid = train_split[train_split[valid_col] == 1].copy()
    val_valid = val_split[val_split[valid_col] == 1].copy()
    test_valid = test_split[test_split[valid_col] == 1].copy()

    print(
        f"Train={len(train_valid)}, "
        f"Validation={len(val_valid)}, "
        f"Test={len(test_valid)}"
    )

    # Train chronological LSTM
    lstm = LSTMForecasterWrapper(
        hidden_dim=64,
        seq_len=14,
        lr=0.002,
        batch_size=64,
        epochs=40,
        patience=6,
        random_state=SEED,
        horizon=horizon,
    )

    print(f"\nTraining chronological LSTM H{horizon}...")

    lstm.fit(
        full_df=df,
        train_split=train_split,
        val_split=val_split,
    )

    checkpoint_path = os.path.join(
        MODELS_DIR,
        f"LSTM_H{horizon}.pt",
    )

    lstm.save_checkpoint(checkpoint_path)

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Best epoch: {lstm.best_epoch}")
    print(f"Best validation loss: {lstm.best_val_loss:.6f}")
    print(
        f"Training duration: "
        f"{lstm.training_duration_sec:.2f} seconds"
    )

    # Predictions
    val_preds = lstm.predict(val_valid)
    test_preds = lstm.predict(test_valid)

    y_val = val_valid[target_col].to_numpy(dtype=float)
    y_test = test_valid[target_col].to_numpy(dtype=float)

    # Static conformal
    static_cp = SplitConformalPredictor(alpha=0.10)
    static_cp.calibrate(y_val, val_preds)

    # Rolling conformal
    val_scores = np.abs(y_val - val_preds)

    rolling_cp = RollingConformalPredictor(
        alpha=0.10,
        window_size=120,
        horizon=horizon,
        mode="pooled",
    )

    roll_lower, roll_upper, _ = rolling_cp.run_rolling_evaluation(
        test_valid,
        test_preds,
        y_test,
        val_scores,
        val_valid["City"].to_numpy(),
    )

    # Random reference benchmark
    all_valid = df[df[valid_col] == 1].copy()

    rng = np.random.RandomState(SEED)
    shuffled = rng.permutation(len(all_valid))

    n_random_test = len(test_valid)

    random_test_idx = shuffled[:n_random_test]
    random_remaining_idx = shuffled[n_random_test:]

    random_test = all_valid.iloc[random_test_idx].copy()
    random_remaining = all_valid.iloc[random_remaining_idx].copy()

    # Split remaining random data into random training and calibration
    rng2 = np.random.RandomState(SEED)

    rem_perm = rng2.permutation(len(random_remaining))

    n_random_val = min(
        len(val_valid),
        max(1, int(0.20 * len(random_remaining))),
    )

    random_val_idx = rem_perm[:n_random_val]
    random_train_idx = rem_perm[n_random_val:]

    random_val = random_remaining.iloc[random_val_idx].copy()
    random_train = random_remaining.iloc[random_train_idx].copy()

    print(
        f"\nRandom reference: "
        f"train={len(random_train)}, "
        f"calibration={len(random_val)}, "
        f"test={len(random_test)}"
    )

    random_lstm = LSTMForecasterWrapper(
        hidden_dim=64,
        seq_len=14,
        lr=0.002,
        batch_size=64,
        epochs=40,
        patience=6,
        random_state=SEED,
        horizon=horizon,
    )

    print(f"Training random-reference LSTM H{horizon}...")

    random_lstm.fit(
        full_df=df,
        train_split=random_train,
        val_split=random_val,
    )

    random_preds = random_lstm.predict(random_test)

    y_random = random_test[target_col].to_numpy(dtype=float)

    random_metrics = calculate_point_metrics(
        y_random,
        random_preds,
    )

    reference_mae = random_metrics["MAE"]

    # Independent random-reference conformal calibration
    random_val_preds = random_lstm.predict(random_val)
    random_y_val = random_val[target_col].to_numpy(dtype=float)

    random_cp = SplitConformalPredictor(alpha=0.10)
    random_cp.calibrate(
        random_y_val,
        random_val_preds,
    )

    # Construct shift slices
    slices = {
        "Chronological_Full": test_valid,
    }

    slices.update(
        ShiftScenarioPartitioner.get_year_slices(test_valid)
    )

    slices.update(
        ShiftScenarioPartitioner.get_seasonal_slices(test_valid)
    )

    slices.update(
        ShiftScenarioPartitioner.get_event_slices(test_valid)
    )

    slices.update(
        ShiftScenarioPartitioner.get_city_slices(test_valid)
    )

    records = []

    # Random reference row first
    rand_lower, rand_upper = random_cp.predict_intervals(
        random_preds
    )

    rand_interval = calculate_interval_metrics(
        y_random,
        rand_lower,
        rand_upper,
        nominal=0.90,
    )

    rand_severity = calculate_severity_metrics(
        y_random,
        random_preds,
        severe_threshold=400.0,
    )

    records.append(
        create_record(
            horizon=horizon,
            scenario="Random_Reference",
            y_true=y_random,
            preds=random_preds,
            static_lower=rand_lower,
            static_upper=rand_upper,
            rolling_lower=rand_lower,
            rolling_upper=rand_upper,
            reference_mae=reference_mae,
            lstm=lstm,
        )
    )

    # Chronological shift slices
    for scenario, slice_df in slices.items():
        positions = test_valid.index.get_indexer(
            slice_df.index
        )

        preds = test_preds[positions]
        y_true = slice_df[target_col].to_numpy(dtype=float)

        static_lower, static_upper = (
            static_cp.predict_intervals(preds)
        )

        rolling_lower = roll_lower[positions]
        rolling_upper = roll_upper[positions]

        records.append(
            create_record(
                horizon=horizon,
                scenario=scenario,
                y_true=y_true,
                preds=preds,
                static_lower=static_lower,
                static_upper=static_upper,
                rolling_lower=rolling_lower,
                rolling_upper=rolling_upper,
                reference_mae=reference_mae,
                lstm=lstm,
            )
        )

    return pd.DataFrame(records)


def create_record(
    horizon,
    scenario,
    y_true,
    preds,
    static_lower,
    static_upper,
    rolling_lower,
    rolling_upper,
    reference_mae,
    lstm,
):
    point = calculate_point_metrics(
        y_true,
        preds,
    )

    static_metrics = calculate_interval_metrics(
        y_true,
        static_lower,
        static_upper,
        nominal=0.90,
    )

    rolling_metrics = calculate_interval_metrics(
        y_true,
        rolling_lower,
        rolling_upper,
        nominal=0.90,
    )

    severity = calculate_severity_metrics(
        y_true,
        preds,
        severe_threshold=400.0,
    )

    return {
        "Horizon": f"H{horizon}",
        "Scenario": scenario,
        "Model": "LSTM",

        "MAE": point["MAE"],
        "RMSE": point["RMSE"],
        "R2": point["R2"],

        "ShiftPenalty": calculate_shift_penalty(
            point["MAE"],
            reference_mae,
        ),

        "SampleCount": point["Count"],

        "PICP_Static": static_metrics["PICP"],
        "MPIW_Static": static_metrics["MPIW"],
        "CoverageGap_Static": static_metrics["CoverageGap"],

        "PICP_Rolling": rolling_metrics["PICP"],
        "MPIW_Rolling": rolling_metrics["MPIW"],
        "CoverageGap_Rolling": rolling_metrics[
            "CoverageGap"
        ],

        "Severe_Recall": severity.get(
            "Severe_Recall",
            np.nan,
        ),

        "Severe_Precision": severity.get(
            "Severe_Precision",
            np.nan,
        ),

        "Severe_F1": severity.get(
            "Severe_F1",
            np.nan,
        ),

        "MAE_Severe": severity.get(
            "MAE_Severe",
            np.nan,
        ),

        "MAE_VeryPoor": severity.get(
            "MAE_VeryPoor",
            np.nan,
        ),

        "MAE_Poor": severity.get(
            "MAE_Poor",
            np.nan,
        ),

        "MAE_Moderate": severity.get(
            "MAE_Moderate",
            np.nan,
        ),

        # For point LSTM predictions, median pinball loss
        # is evaluated on the point forecast.
        # Lower/upper conformal limits provide the
        # corresponding tail-loss estimates.
        "Pinball_0.05": calculate_pinball_loss(
            y_true,
            static_lower,
            0.05,
        ),

        "Pinball_0.50": calculate_pinball_loss(
            y_true,
            preds,
            0.50,
        ),

        "Pinball_0.95": calculate_pinball_loss(
            y_true,
            static_upper,
            0.95,
        ),

        "Training_Epochs": lstm.best_epoch,
        "Training_Duration_Sec": (
            lstm.training_duration_sec
        ),
        "Best_Val_Loss": lstm.best_val_loss,
    }


def merge_with_master(lstm_df):
    if not os.path.exists(MASTER_RESULTS_PATH):
        print(
            "Master metrics file not found. "
            "Skipping automatic merge."
        )
        return

    master = pd.read_csv(MASTER_RESULTS_PATH)

    # Prevent duplicate LSTM rows
    master = master[
        master["Model"] != "LSTM"
    ].copy()

    # metrics_all_horizons.csv currently contains
    # the common evaluation columns only.
    common_cols = [
        col
        for col in master.columns
        if col in lstm_df.columns
    ]

    combined = pd.concat(
        [
            master,
            lstm_df[common_cols],
        ],
        ignore_index=True,
    )

    combined.to_csv(
        MASTER_RESULTS_PATH,
        index=False,
    )

    print(
        f"Updated master results: "
        f"{MASTER_RESULTS_PATH}"
    )
    print(
        f"Final master rows: {len(combined)}"
    )


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    print(
        f"Loading features from {FEATURES_PATH}"
    )

    df = pd.read_csv(FEATURES_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    print(
        f"Loaded {len(df)} rows."
    )

    h1_results = evaluate_lstm_horizon(
        df,
        horizon=1,
    )

    h3_results = evaluate_lstm_horizon(
        df,
        horizon=3,
    )

    lstm_results = pd.concat(
        [
            h1_results,
            h3_results,
        ],
        ignore_index=True,
    )

    lstm_results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("LSTM EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Saved LSTM results to {RESULTS_PATH}"
    )

    print(
        "\nChronological results:"
    )

    print(
        lstm_results[
            lstm_results["Scenario"]
            == "Chronological_Full"
        ][
            [
                "Horizon",
                "MAE",
                "RMSE",
                "R2",
                "Training_Epochs",
                "Best_Val_Loss",
            ]
        ].to_string(index=False)
    )

    merge_with_master(lstm_results)


if __name__ == "__main__":
    main()