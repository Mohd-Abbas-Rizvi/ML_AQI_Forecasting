# AQI Forecasting Under Distribution Shift — Project Plan and Implementation Status

## 1. Project Objective

The goal of this project is to study the reliability of machine learning models for AQI forecasting when the data distribution changes across time, seasons, pollution events, and geographic regions.

The project addresses both:

1. **Forecast accuracy**
2. **Prediction uncertainty and calibration**

Two forecasting horizons are studied:

```text
H1 = next-day AQI
H3 = three-day-ahead AQI
```

The primary experiments use five Indian cities:

```text
Bengaluru
Chennai
Delhi
Hyderabad
Lucknow
```

A secondary UCI Beijing PM2.5 experiment evaluates whether the conclusions generalize beyond the primary Indian AQI dataset.

---

# 2. Core Research Questions

## RQ1 — Forecast Accuracy Under Shift

How much does forecasting error change when the model encounters a distribution different from its reference distribution?

Measured using:

```text
MAE
RMSE
R²
Shift Penalty
```

with:

$$
\text{Shift Penalty}
=
\frac{\text{MAE}_{shift}}
{\text{MAE}_{reference}}
$$

---

## RQ2 — Prediction Interval Reliability

Do nominal 90% conformal prediction intervals retain approximately 90% empirical coverage under temporal, seasonal, event-driven, and geographic shift?

Methods:

```text
Static Split Conformal
Rolling Adaptive Conformal
```

Metrics:

```text
PICP
MPIW
Coverage Gap
```

---

## RQ3 — Severe Pollution Forecasting

Can the models correctly identify extreme AQI episodes that are important for public-health warnings?

Severe pollution threshold:

$$
AQI > 400
$$

Metrics:

```text
Severe Recall
Severe Precision
Severe F1
Severity-Bucket MAE
```

---

# 3. Primary Dataset

Dataset:

```text
Air Quality Data in India
```

Selected cities:

```text
Bengaluru
Chennai
Delhi
Hyderabad
Lucknow
```

Calendar records after continuity enforcement:

```text
10,042
```

Approximate period:

```text
2015 to mid-2020
```

The original target AQI is preserved.

Future target values are never backfilled or artificially interpolated.

---

# 4. Secondary Dataset

Dataset:

```text
UCI Beijing PM2.5
```

Original hourly rows:

```text
43,824
```

Daily aggregated records:

```text
1,826
```

Period:

```text
2010–2014
```

The experiment evaluates temporal transfer from an earlier training regime to later shifted years.

---

# 5. Leakage-Safe Preprocessing

Implemented in:

```text
src/data/
src/features/
```

The pipeline:

- enforces chronological ordering,
- verifies daily continuity,
- creates H1 and H3 future targets,
- preserves unmodified observed targets,
- fits imputation statistics only on training-period data,
- uses causal forward filling,
- generates lag features,
- generates rolling statistics,
- generates calendar/cyclical features,
- generates event indicators,
- and excludes metadata/target columns from predictive inputs.

Three preprocessing variants are evaluated:

```text
ffill_train_median
train_season_median
causal_rolling_mean
```

Primary processed dataset:

```text
data/processed/features_ffill_train_median.csv
```

---

# 6. Model Suite

## 6.1 Baselines

Implemented in:

```text
src/models/baselines.py
```

Models:

```text
Persistence
Seasonal Naive
```

---

## 6.2 From-Scratch NumPy Models

Implemented in:

```text
src/models/scratch_numpy.py
```

Models:

```text
NumPy Ridge
NumPy MLP
```

---

## 6.3 Classical Models

Implemented in:

```text
src/models/classical.py
```

Models:

```text
Ridge
Random Forest
Gradient Boosting
XGBoost
```

---

## 6.4 Feed-Forward Neural Model

Implemented in:

```text
src/models/neural.py
```

Model:

```text
PyTorch MLP
```

---

## 6.5 LSTM Sequence Model

Implemented in:

```text
src/models/lstm.py
```

Reproduction runner:

```text
experiments/run_lstm.py
```

Configuration:

```text
Sequence length: 14
Features per time step: 18
Hidden dimension: 64
LSTM layers: 1
Dropout: 0.1
Optimizer: Adam
Loss: SmoothL1 / Huber
Maximum epochs: 40
Early stopping patience: 6
```

Checkpoint files:

```text
models/LSTM_H1.pt
models/LSTM_H3.pt
```

Final reproduced chronological performance:

```text
H1:
MAE  = 17.7555
RMSE = 26.9446
R²   = 0.9096
Best epoch = 5

H3:
MAE  = 30.9862
RMSE = 47.5247
R²   = 0.7134
Best epoch = 9
```

---

## 6.6 Quantile Model

Implemented in:

```text
src/uncertainty/quantile.py
```

Model:

```text
Quantile Gradient Boosting
```

Quantiles:

$$
0.05,\ 0.50,\ 0.95
$$

Evaluation includes pinball loss.

---

# 7. Distribution-Shift Scenarios

The evaluation covers:

## Reference

```text
Random_Reference
```

## Chronological

```text
Chronological_Full
```

## Annual

```text
Year_2019
Year_2020
```

## Seasonal

```text
Season_Winter
Season_Summer
Season_Monsoon
Season_PostMonsoon
```

## Event-Driven

```text
Event_Diwali
Event_StubbleBurning
Event_Lockdown
Event_WinterInversion
```

## Geographic

```text
City_Bengaluru
City_Chennai
City_Delhi
City_Hyderabad
City_Lucknow
```

Total:

```text
17 scenarios per horizon
```

---

# 8. Evaluation Metrics

## Point Forecast Metrics

```text
MAE
RMSE
R²
```

## Distribution-Shift Metric

```text
Shift Penalty
```

## Conformal Metrics

```text
PICP
MPIW
Coverage Gap
```

## Quantile Metrics

```text
Pinball Loss at 0.05
Pinball Loss at 0.50
Pinball Loss at 0.95
```

## Severe Pollution Metrics

```text
Severe Recall
Severe Precision
Severe F1
MAE_Moderate
MAE_Poor
MAE_VeryPoor
MAE_Severe
```

---

# 9. Ablation Studies

Implemented in:

```text
experiments/run_ablations.py
```

Studies include:

1. Imputation strategy comparison
2. AQI lag-only features
3. Pollutant feature contribution
4. Calendar feature contribution
5. Event indicator contribution
6. Conformal rolling-window sensitivity
7. Spatial held-out-city transfer

Important outputs:

```text
results/tables/ablation_imputation.csv
results/tables/ablation_features.csv
results/tables/ablation_conformal_window.csv
results/tables/spatial_held_out_city.csv
```

---

# 10. Additional Robustness Experiments

Additional experiment scripts include:

```text
experiments/run_stability_analysis.py
experiments/run_held_out_city.py
experiments/run_seasonal_holdout.py
```

These support:

- temporal stability analysis,
- model variation analysis,
- held-out-city transfer,
- seasonal calibration assessment,
- and statistical comparison against Persistence.

---

# 11. LSTM Integration

The LSTM is intentionally evaluated using a separate runner because it consumes:

```text
(batch, 14 days, 18 features)
```

rather than the flat tabular feature representation used by the other models.

Correct workflow:

```text
shift_evaluator.py
        ↓
340 tabular-model scenario rows
        ↓
run_lstm.py
        ↓
34 LSTM scenario rows
        ↓
metrics_all_horizons.csv
        ↓
374 total rows
```

This separation avoids forcing incompatible tabular and sequence-model interfaces into a single evaluator.

---

# 12. Main Results

## H1

Persistence:

```text
MAE = 21.06
```

Best model:

```text
Quantile GBM
MAE = 16.73
Improvement ≈ 20.6%
```

LSTM:

```text
MAE = 17.76
Improvement ≈ 15.7%
```

---

## H3

Persistence:

```text
MAE = 34.56
```

Best MAE:

```text
Ridge / NumPy Ridge
MAE = 29.95
Improvement ≈ 13.3%
```

LSTM:

```text
MAE = 30.99
Improvement ≈ 10.3%
```

---

# 13. Secondary Beijing Benchmark

Implemented in:

```text
src/secondary/beijing_pipeline.py
```

Outputs:

```text
data/processed/beijing_daily_features.csv
results/tables/beijing_shift_results.csv
```

The experiment evaluates:

```text
Persistence
Ridge
Gradient Boosting
Random Forest
```

for:

```text
H1
H3
```

under a shifted later-year test period.

---

# 14. Visualization Suite

Implemented in:

```text
src/visualization/plots.py
```

Final figures:

```text
fig01_missing_values.png
fig02_aqi_trends_by_city.png
fig03_model_accuracy_comparison.png
fig04_shift_penalty_heatmap.png
fig05_conformal_coverage_and_width.png
fig06_severe_alert_and_bucket_errors.png
fig07_delhi_spike_conformal_series.png
```

LSTM is included in the overall model-comparison, shift-penalty, and severe-pollution comparisons through the final master metrics table.

Gradient Boosting remains the intentional focused case study in the conformal and Delhi extreme-event figures.

---

# 15. Jupyter Notebooks

## Complete Project Notebook

```text
notebooks/ML_AQI_Complete_Project.ipynb
```

Contains:

- project overview,
- dataset inspection,
- preprocessing,
- feature definitions,
- model descriptions,
- LSTM architecture/checkpoint inspection,
- metrics,
- result tables,
- and project conclusions.

## Results Notebook

```text
notebooks/ML_AQI_Results.ipynb
```

Contains:

- model comparisons,
- seasonal/year analysis,
- shift-penalty heatmaps,
- severe-pollution analysis,
- and final experimental plots.

Both notebooks use:

```text
Python (ML AQI Project)
```

with the project-specific `.venv`.

---

# 16. Testing

Test files:

```text
tests/test_leakage_and_splits.py
tests/test_lstm.py
tests/test_methodology_corrections.py
tests/test_pipeline_reproducibility.py
```

Verified result:

```text
24 passed
```

The tests cover:

- temporal ordering,
- H1/H3 target alignment,
- leakage protection,
- feature sanitization,
- conformal behavior,
- sequence construction,
- city-boundary isolation,
- LSTM training,
- and LSTM checkpoint restoration.

---

# 17. Final Reproduction Sequence

From the project root:

```bash
source .venv/bin/activate
```

Then execute:

```bash
PYTHONPATH=. python src/data/preprocess.py
```

```bash
PYTHONPATH=. python src/evaluation/shift_evaluator.py
```

```bash
PYTHONPATH=. python experiments/run_lstm.py
```

```bash
PYTHONPATH=. python experiments/run_ablations.py
```

```bash
PYTHONPATH=. python src/secondary/beijing_pipeline.py
```

```bash
PYTHONPATH=. python src/visualization/plots.py
```

Finally:

```bash
pytest -q
```

Expected:

```text
24 passed
```

---

# 18. Project Directory

Final recommended structure:

```text
ML_Project/
├── README.md
├── PROJECT_PLAN.md
├── REPORT.md
├── requirements.txt
├── pytest.ini
├── app.py
├── AQI_Forecasting_Poster-1.pdf
│
├── data/
├── src/
├── experiments/
├── models/
├── notebooks/
├── results/
└── tests/
```

Local-only directories such as:

```text
.venv/
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
```

must not be committed to GitHub.

---

# 19. Reproducibility Status

Current local verification status:

```text
Preprocessing                 PASS
Tabular evaluation            PASS
LSTM H1/H3 reproduction       PASS
Ablation experiments          PASS
Beijing secondary experiment  PASS
Figure generation             PASS
Complete notebook             PASS
Results notebook              PASS
Streamlit application         PASS
Unit tests                    24/24 PASS
```

The LSTM H1/H3 rerun reproduced the previously stored results.

---

# 20. Remaining Final Submission Work

Before final course submission:

- finalize `.gitignore`,
- verify Streamlit uses the final LSTM checkpoint interface,
- finalize GitHub repository,
- record Git commit hashes and timestamps,
- finalize reproducibility declaration,
- prepare CVPR-format 8-page report,
- and provide the final GitHub repository URL.

No major model or experiment redesign is currently required.