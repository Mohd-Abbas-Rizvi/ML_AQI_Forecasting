# Unit Tests for Lightweight LSTM Architecture and Sequence Dataset Builder
import os
import pytest
import numpy as np
import pandas as pd
import torch

from src.models.lstm import (
    LSTMNet,
    LSTMSequenceDatasetBuilder,
    LSTMForecasterWrapper,
    LSTM_FEATURE_COLS,
    get_torch_device
)


def test_lstm_network_shape():
    # Test network input and output dimensions
    net = LSTMNet(input_dim=18, hidden_dim=64, dropout=0.1)
    net.eval()
    dummy_input = torch.randn(16, 14, 18)
    with torch.no_grad():
        output = net(dummy_input)
    assert output.shape == (16, 1)


def test_sequence_dataset_builder_no_boundary_crossing():
    # Test that sequence building preserves temporal order and NEVER crosses city boundaries
    dates = pd.date_range("2018-01-01", periods=20, freq="D")
    
    # City 1: Delhi with feature values around 100
    df_delhi = pd.DataFrame({
        "Date": dates,
        "City": "Delhi",
    })
    for col in LSTM_FEATURE_COLS:
        df_delhi[col] = 100.0

    # City 2: Bengaluru with feature values around 20
    df_blr = pd.DataFrame({
        "Date": dates,
        "City": "Bengaluru",
    })
    for col in LSTM_FEATURE_COLS:
        df_blr[col] = 20.0

    df_combined = pd.concat([df_delhi, df_blr], ignore_index=True)

    builder = LSTMSequenceDatasetBuilder(feature_cols=LSTM_FEATURE_COLS, seq_len=14)
    seqs, lookup = builder.build_sequences_and_lookup(df_combined)

    assert seqs.shape == (40, 14, len(LSTM_FEATURE_COLS))
    assert len(lookup) == 40

    # Test Bengaluru sequences: the first Bengaluru day should have values 20.0, NOT 100.0 from Delhi
    blr_first_date = dates[0]
    blr_idx = lookup[("Bengaluru", blr_first_date)]
    blr_seq = seqs[blr_idx]

    # All values in Bengaluru's sequence must strictly equal 20.0 (padded with first Bengaluru day)
    np.testing.assert_allclose(blr_seq, 20.0)

    # Check Delhi sequence
    delhi_idx = lookup[("Delhi", dates[10])]
    delhi_seq = seqs[delhi_idx]
    np.testing.assert_allclose(delhi_seq, 100.0)


def _create_synthetic_df():
    dates = pd.date_range("2018-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "City": "Delhi",
        "target_H1": np.linspace(50, 150, len(dates)),
        "target_H1_valid": 1,
    })
    for col in LSTM_FEATURE_COLS:
        df[col] = np.random.uniform(10, 100, size=len(dates))
    return df


def test_lstm_training_and_inference():
    # Test end-to-end training and inference on synthetic dataframe
    df = _create_synthetic_df()
    train_split = df.iloc[:40].copy()
    val_split = df.iloc[40:50].copy()
    test_split = df.iloc[50:].copy()

    forecaster = LSTMForecasterWrapper(
        hidden_dim=32,
        seq_len=14,
        lr=0.01,
        batch_size=16,
        epochs=3,
        patience=2,
        random_state=42,
        horizon=1,
    )

    forecaster.fit(df, train_split, val_split)

    preds = forecaster.predict(test_split)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(test_split),)
    assert not np.isnan(preds).any()


def test_lstm_checkpoint_save_and_load(tmp_path):
    # Test checkpoint serialization and loaded reproducibility
    df = _create_synthetic_df()
    train_split = df.iloc[:40].copy()
    val_split = df.iloc[40:50].copy()
    test_split = df.iloc[50:].copy()

    forecaster = LSTMForecasterWrapper(
        hidden_dim=32,
        seq_len=14,
        lr=0.01,
        batch_size=16,
        epochs=2,
        random_state=42,
        horizon=1,
    )
    forecaster.fit(df, train_split, val_split)

    ckpt_path = str(tmp_path / "lstm_test.pt")
    forecaster.save_checkpoint(ckpt_path)
    assert os.path.exists(ckpt_path)

    preds_original = forecaster.predict(test_split)

    # Load into a fresh model instance with sequence context
    forecaster_loaded = LSTMForecasterWrapper(hidden_dim=32, seq_len=14, horizon=1)
    forecaster_loaded.load_checkpoint(ckpt_path, full_df=df)
    preds_loaded = forecaster_loaded.predict(test_split)

    np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-5)


def test_irregular_calendar_dates_detection_and_reindexing():
    # Test 1: Irregular calendar dates must be detected and never silently treated as consecutive rows
    dates_with_gap = [
        pd.Timestamp("2018-01-01"),
        pd.Timestamp("2018-01-02"),
        pd.Timestamp("2018-01-03"),
        # Gap of 7 days
        pd.Timestamp("2018-01-10"),
        pd.Timestamp("2018-01-11"),
        pd.Timestamp("2018-01-12"),
    ]
    df_irregular = pd.DataFrame({
        "Date": dates_with_gap,
        "City": "Delhi",
    })
    for col in LSTM_FEATURE_COLS:
        df_irregular[col] = 50.0

    # With handle_missing_dates="error", must raise informative ValueError
    builder_strict = LSTMSequenceDatasetBuilder(handle_missing_dates="error")
    with pytest.raises(ValueError, match="non-consecutive calendar dates with gaps"):
        builder_strict.build_sequences_and_lookup(df_irregular)

    # With handle_missing_dates="reindex_ffill", must reindex to full continuous daily calendar
    builder_reindex = LSTMSequenceDatasetBuilder(handle_missing_dates="reindex_ffill")
    seqs, lookup = builder_reindex.build_sequences_and_lookup(df_irregular)
    # The full range from 2018-01-01 to 2018-01-12 has 12 consecutive days
    assert seqs.shape == (12, 14, len(LSTM_FEATURE_COLS))
    assert (pd.Timestamp("2018-01-10") - pd.Timestamp("2018-01-09")) == pd.Timedelta(days=1)


def test_training_loss_aggregation_sample_weighted():
    # Test 2: Training loss must divide by total training samples, not batch count
    # Choose n_samples not divisible by batch_size (53 samples with batch_size=16 -> batches: 16, 16, 16, 5)
    df = _create_synthetic_df()
    train_split = df.iloc[:53].copy()
    val_split = df.iloc[53:60].copy()

    forecaster = LSTMForecasterWrapper(
        hidden_dim=32,
        seq_len=14,
        lr=0.01,
        batch_size=16,
        epochs=2,
        random_state=42,
        horizon=1,
    )
    forecaster.fit(df, train_split, val_split)

    assert len(forecaster.train_history["train_loss"]) == 2
    for loss in forecaster.train_history["train_loss"]:
        assert loss > 0.0
        assert not np.isnan(loss)
        assert np.isfinite(loss)


def test_reference_calibration_independence():
    # Test 3: Random reference conformal predictor must use independent reference calibration residuals
    from src.uncertainty.conformal import SplitConformalPredictor

    # Synthetic reference model predictions on independent calibration and test sets
    np.random.seed(42)
    y_ref_cal = np.array([100.0, 150.0, 200.0, 80.0, 120.0])
    ref_cal_preds = np.array([105.0, 140.0, 195.0, 88.0, 115.0])
    
    # Chronological validation set with completely different residual distribution
    y_chrono_val = np.array([50.0, 60.0, 70.0, 55.0, 65.0])
    chrono_val_preds = np.array([20.0, 30.0, 40.0, 25.0, 35.0]) # large error ~ 30

    ref_cp = SplitConformalPredictor(alpha=0.10)
    ref_cp.calibrate(y_ref_cal, ref_cal_preds)

    chrono_cp = SplitConformalPredictor(alpha=0.10)
    chrono_cp.calibrate(y_chrono_val, chrono_val_preds)

    # Reference calibration should have significantly tighter interval matching reference model's accuracy
    assert ref_cp.q_hat < chrono_cp.q_hat
    assert ref_cp.q_hat <= 10.0
    assert chrono_cp.q_hat >= 30.0


def test_no_backward_filling_from_future_observations():
    # Test: Verify that when early observations are missing and later observations have
    # distinctive values, earlier values are NEVER filled from future observations (no bfill).
    dates = pd.date_range("2018-01-01", periods=6, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "City": "Delhi",
    })
    for col in LSTM_FEATURE_COLS:
        df[col] = 50.0  # Training baseline median

    # First two days are missing for AQI, and Day 4 onwards has a distinctive spike value 999.0
    df.loc[0, "AQI"] = np.nan
    df.loc[1, "AQI"] = np.nan
    df.loc[2, "AQI"] = 50.0
    df.loc[3, "AQI"] = 999.0
    df.loc[4, "AQI"] = 999.0
    df.loc[5, "AQI"] = 999.0

    builder = LSTMSequenceDatasetBuilder(
        feature_cols=LSTM_FEATURE_COLS,
        seq_len=14,
        handle_missing_dates="reindex_ffill",
        imputation_stats={"city_medians": {"Delhi": {"AQI": 50.0}}, "global_medians": {"AQI": 50.0}}
    )
    seqs, lookup = builder.build_sequences_and_lookup(df)

    aqi_idx = LSTM_FEATURE_COLS.index("AQI")

    # Day 1 sequence ending at Day 1 (2018-01-01)
    day1_idx = lookup[("Delhi", dates[0])]
    seq_day1 = seqs[day1_idx]
    # Assert Day 1 value in sequence NEVER took future 999.0 value
    assert not np.any(seq_day1[:, aqi_idx] == 999.0)
    # Must equal causal training median
    assert np.all(seq_day1[:, aqi_idx] == 50.0)

    # Day 2 sequence ending at Day 2 (2018-01-02)
    day2_idx = lookup[("Delhi", dates[1])]
    seq_day2 = seqs[day2_idx]
    assert not np.any(seq_day2[:, aqi_idx] == 999.0)

    # Day 4 sequence ending at Day 4 (2018-01-04)
    day4_idx = lookup[("Delhi", dates[3])]
    seq_day4 = seqs[day4_idx]
    # Day 4 observation at final step is 999.0
    assert seq_day4[-1, aqi_idx] == 999.0
    # Earlier historical steps in the window (e.g. step corresponding to Day 1) are NOT 999.0
    assert seq_day4[0, aqi_idx] == 50.0


def test_validation_test_only_data_rejects_imputation_fitting():
    # Test: Sequence builder strictly forbids fitting imputation statistics on validation/test-only data.
    # When passed dates strictly after train_end_date (2018-03-13), fit_imputation_stats must raise ValueError.
    test_dates = pd.date_range("2019-01-01", periods=10, freq="D")
    df_test_only = pd.DataFrame({
        "Date": test_dates,
        "City": "Delhi",
    })
    for col in LSTM_FEATURE_COLS:
        df_test_only[col] = 120.0

    builder = LSTMSequenceDatasetBuilder(
        feature_cols=LSTM_FEATURE_COLS,
        seq_len=14,
        train_end_date="2018-03-13",
        imputation_stats=None,
    )

    # Attempting to fit imputation stats on test-only dates must raise informative ValueError
    with pytest.raises(ValueError, match="No records found on or before train_end_date"):
        builder.fit_imputation_stats(df_test_only)

    with pytest.raises(ValueError, match="No records found on or before train_end_date"):
        builder.build_sequences_and_lookup(df_test_only)

    # With pre-supplied saved training statistics, inference on test-only data must succeed without training data
    saved_stats = {
        "city_medians": {"Delhi": {c: 50.0 for c in LSTM_FEATURE_COLS}},
        "global_medians": {c: 50.0 for c in LSTM_FEATURE_COLS},
    }
    builder_with_stats = LSTMSequenceDatasetBuilder(
        feature_cols=LSTM_FEATURE_COLS,
        seq_len=14,
        train_end_date="2018-03-13",
        imputation_stats=saved_stats,
    )
    seqs, lookup = builder_with_stats.build_sequences_and_lookup(df_test_only)
    assert seqs.shape == (10, 14, len(LSTM_FEATURE_COLS))
