# AQI Forecasting Under Distribution Shift: How Accurate and How Well Calibrated?

An empirical machine learning project for forecasting Air Quality Index (AQI) under temporal, seasonal, event-driven, and spatial distribution shifts.

The project evaluates both **forecast accuracy** and **uncertainty calibration** for next-day ($H=1$) and 3-day-ahead ($H=3$) AQI prediction across five Indian cities. It also includes a secondary validation experiment using the UCI Beijing PM2.5 dataset.

## Authors

- **Mohd Abbas Rizvi**
- **Hatim Bharmal**

**Academic Affiliation:** M.Tech CSE / Artificial Intelligence, Indian Institute of Technology Jodhpur

**Project Poster:** `AQI_Forecasting_Poster-1.pdf`

---

# 1. Research Questions

The project investigates three main questions:

1. **How much does AQI forecasting performance degrade under distribution shift?**

   Shift is evaluated across:
   - chronological test periods,
   - different years,
   - seasons,
   - pollution events,
   - lockdown periods,
   - and different cities.

   1. The main shift measure is:

$$
\text{Shift Penalty}
=
\frac{\text{MAE}_{\text{shifted}}}
{\text{MAE}_{\text{reference}}}
$$

2. **Do nominal 90% prediction intervals remain calibrated under distribution shift?**

   Static split-conformal prediction is compared against adaptive rolling conformal recalibration.

3. **Can machine learning models provide practically useful forecasts during severe pollution episodes?**

   Models are evaluated using conventional regression metrics as well as severe-event recall, precision, F1, and AQI severity-bucket errors.

---

# 2. Datasets

## 2.1 Indian AQI Dataset

Primary experiments use the **Air Quality Data in India** dataset.

Five cities are studied:

- Bengaluru
- Chennai
- Delhi
- Hyderabad
- Lucknow

The selected data span approximately:

```text
2015-01-01 to 2020-07-01
```

After calendar continuity processing, the combined five-city dataset contains:

```text
10,042 daily city records
```

The primary target is daily AQI.

The project preserves raw observed targets and does not create synthetic target values through backward interpolation.

## 2.2 UCI Beijing PM2.5 Dataset

A secondary benchmark uses the UCI Beijing PM2.5 dataset.

Original data:

```text
43,824 hourly observations
```

After daily aggregation:

```text
1,826 days
2010-01-01 to 2014-12-31
```

The secondary experiment uses earlier years for training and later years as a shifted test period.

---

# 3. Forecast Horizons

Two forecasting horizons are evaluated:

- **H1:** next-day AQI prediction
- **H3:** three-day-ahead AQI prediction

Targets are created using calendar-aligned future observations.

Missing future targets are excluded from evaluation rather than synthetically imputed.

---

# 4. Models

The project evaluates **11 model families**.

## Baselines

1. Persistence
2. Seasonal Naive

## From-Scratch NumPy Models

3. NumPy Ridge Regression
4. NumPy MLP

## Classical Machine Learning

5. Ridge Regression
6. Random Forest
7. Gradient Boosting
8. XGBoost

## Neural Models

9. PyTorch MLP
10. Lightweight PyTorch LSTM

## Probabilistic / Quantile Model

11. Quantile Gradient Boosting

The Quantile GBM estimates:

$$
\alpha \in \{0.05, 0.50, 0.95\}
$$

using pinball loss.

---

# 5. Lightweight LSTM

The LSTM is implemented separately from the tabular model pipeline because it consumes temporal sequences rather than a single engineered feature vector.

Architecture:

```text
Sequence length: 14 days
Features per day: 18
LSTM layers: 1
Hidden dimension: 64
Head: Linear(64 -> 32) -> ReLU -> Dropout -> Linear(32 -> 1)
Loss: Smooth L1 / Huber
Optimizer: Adam
Early stopping patience: 6
```

The 18 daily input features include:

```text
AQI
PM2.5
PM10
NO2
SO2
CO
O3
sin_DayOfYear
cos_DayOfYear
sin_Month
cos_Month
sin_DayOfWeek
cos_DayOfWeek
is_weekend
is_diwali_window
is_stubble_burning
is_winter_inversion
is_lockdown
```

The sequence builder:

- never crosses city boundaries,
- uses true daily temporal order,
- uses causal forward filling only,
- fits fallback imputation statistics only on the training period,
- and avoids future lookahead.

LSTM checkpoints are stored as:

```text
models/LSTM_H1.pt
models/LSTM_H3.pt
```

Current reproduced chronological results:

| Horizon | MAE | RMSE | R² | Best Epoch |
|---|---:|---:|---:|---:|
| H1 | 17.7555 | 26.9446 | 0.9096 | 5 |
| H3 | 30.9862 | 47.5247 | 0.7134 | 9 |

---

# 6. Project Structure

```text
ML_Project/
│
├── AQI_Forecasting_Poster-1.pdf
├── README.md
├── PROJECT_PLAN.md
├── REPORT.md
├── requirements.txt
├── pytest.ini
├── app.py
│
├── data/
│   ├── raw/
│   │   ├── city_day.csv
│   │   └── PRSA_data_2010.1.1-2014.12.31.csv
│   │
│   └── processed/
│       ├── features_ffill_train_median.csv
│       ├── features_train_season_median.csv
│       ├── features_causal_rolling_mean.csv
│       └── beijing_daily_features.csv
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── imputation.py
│   │   ├── preprocess.py
│   │   └── splitters.py
│   │
│   ├── features/
│   │   └── engineer.py
│   │
│   ├── models/
│   │   ├── baselines.py
│   │   ├── classical.py
│   │   ├── scratch_numpy.py
│   │   ├── neural.py
│   │   └── lstm.py
│   │
│   ├── uncertainty/
│   │   ├── conformal.py
│   │   └── quantile.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── shift_evaluator.py
│   │
│   ├── secondary/
│   │   └── beijing_pipeline.py
│   │
│   └── visualization/
│       └── plots.py
│
├── experiments/
│   ├── run_lstm.py
│   ├── run_ablations.py
│   ├── run_stability_analysis.py
│   ├── run_held_out_city.py
│   └── run_seasonal_holdout.py
│
├── models/
│   ├── Persistence_H1.joblib
│   ├── Persistence_H3.joblib
│   ├── SeasonalNaive_H1.joblib
│   ├── SeasonalNaive_H3.joblib
│   ├── NumPy_Ridge_H1.joblib
│   ├── NumPy_Ridge_H3.joblib
│   ├── NumPy_MLP_H1.joblib
│   ├── NumPy_MLP_H3.joblib
│   ├── Ridge_H1.joblib
│   ├── Ridge_H3.joblib
│   ├── RandomForest_H1.joblib
│   ├── RandomForest_H3.joblib
│   ├── GradientBoosting_H1.joblib
│   ├── GradientBoosting_H3.joblib
│   ├── XGBoost_H1.joblib
│   ├── XGBoost_H3.joblib
│   ├── PyTorch_MLP_H1.joblib
│   ├── PyTorch_MLP_H3.joblib
│   ├── Quantile_GBM_H1.joblib
│   ├── Quantile_GBM_H3.joblib
│   ├── LSTM_H1.pt
│   └── LSTM_H3.pt
│
├── notebooks/
│   ├── ML_AQI_Complete_Project.ipynb
│   └── ML_AQI_Results.ipynb
│
├── results/
│   ├── tables/
│   └── figures/
│
└── tests/
    ├── test_leakage_and_splits.py
    ├── test_lstm.py
    ├── test_methodology_corrections.py
    └── test_pipeline_reproducibility.py
```

---

# 7. Environment Setup

The final local project was verified using:

```text
Python 3.11.17
NumPy 2.4.6
Pandas 3.1.0
scikit-learn 1.9.1
PyTorch 2.15.0
```

Create a dedicated environment:

```bash
cd /path/to/ML_Project

python3.11 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt
```

Verify Python:

```bash
which python
python --version
```

The Python executable should point to:

```text
ML_Project/.venv/bin/python
```

---

# 8. Reproduction Instructions

All commands below must be executed from the project root.

Activate the environment:

```bash
source .venv/bin/activate
```

## Step 1 — Preprocess Data

```bash
PYTHONPATH=. python src/data/preprocess.py
```

This:

- loads the five-city Indian AQI dataset,
- enforces daily calendar continuity,
- generates leakage-safe features,
- creates H1 and H3 targets,
- evaluates three imputation strategies,
- and writes processed datasets and sample-count tables.

Expected main dataset size:

```text
10,042 rows
83 processed columns
74 predictive features in the main tabular evaluation
```

---

## Step 2 — Run Baseline and Tabular Model Evaluation

```bash
PYTHONPATH=. python src/evaluation/shift_evaluator.py
```

This trains and evaluates:

```text
Persistence
SeasonalNaive
NumPy_Ridge
NumPy_MLP
Ridge
RandomForest
GradientBoosting
XGBoost
PyTorch_MLP
Quantile_GBM
```

for both H1 and H3.

It evaluates:

- random-reference performance,
- chronological performance,
- yearly shifts,
- seasonal shifts,
- pollution-event shifts,
- city-specific performance,
- conformal coverage,
- severity metrics,
- and shift penalties.

Main outputs:

```text
results/tables/metrics_H1.csv
results/tables/metrics_H3.csv
results/tables/metrics_all_horizons.csv
```

At this stage the master table contains the tabular-model rows.

---

## Step 3 — Run the LSTM Sequence Model

```bash
PYTHONPATH=. python experiments/run_lstm.py
```

This:

- trains H1 and H3 LSTM models,
- uses 14-day temporal sequences,
- saves the PyTorch checkpoints,
- evaluates all 17 shift scenarios,
- produces static and rolling conformal metrics,
- saves the LSTM-specific table,
- and merges the LSTM rows into the master metrics table.

Outputs:

```text
models/LSTM_H1.pt
models/LSTM_H3.pt
results/tables/lstm_metrics.csv
results/tables/metrics_all_horizons.csv
```

Expected final master-table size:

```text
374 rows
```

The final master table contains:

```text
340 tabular-model rows
+ 34 LSTM rows
= 374 rows
```

---

## Step 4 — Run Ablation Experiments

```bash
PYTHONPATH=. python experiments/run_ablations.py
```

Ablations include:

- alternative imputation strategies,
- AQI lag-only features,
- pollutant features,
- calendar features,
- event indicators,
- conformal rolling-window sensitivity,
- and held-out-city transfer.

Important outputs include:

```text
results/tables/ablation_imputation.csv
results/tables/ablation_features.csv
results/tables/ablation_conformal_window.csv
results/tables/spatial_held_out_city.csv
```

---

## Step 5 — Run Secondary Beijing Experiment

```bash
PYTHONPATH=. python src/secondary/beijing_pipeline.py
```

This processes the UCI Beijing PM2.5 dataset and evaluates H1/H3 forecasting under temporal distribution shift.

Output:

```text
results/tables/beijing_shift_results.csv
```

---

## Step 6 — Generate Publication Figures

```bash
PYTHONPATH=. python src/visualization/plots.py
```

Generated figures:

```text
results/figures/fig01_missing_values.png
results/figures/fig02_aqi_trends_by_city.png
results/figures/fig03_model_accuracy_comparison.png
results/figures/fig04_shift_penalty_heatmap.png
results/figures/fig05_conformal_coverage_and_width.png
results/figures/fig06_severe_alert_and_bucket_errors.png
results/figures/fig07_delhi_spike_conformal_series.png
```

Figures 3, 4, and the severe-event comparison include LSTM results through the final `metrics_all_horizons.csv`.

Figures 5 and 7 are intentionally Gradient Boosting case studies.

---

## Step 7 — Run Tests

```bash
pytest -q
```

Current verified result:

```text
24 passed
```

The test suite checks:

- target/horizon alignment,
- temporal ordering,
- leakage prevention,
- sequence generation,
- city-boundary protection,
- LSTM checkpoint save/load,
- conformal behavior,
- and pipeline reproducibility.

Some newer Pandas versions may emit deprecation warnings; these do not change the numerical results.

---

## Step 8 — Run Jupyter Notebooks

Launch:

```bash
jupyter lab
```

Use the kernel:

```text
Python (ML AQI Project)
```

The kernel should point to:

```text
<project-root>/.venv/bin/python
```

Run:

```text
notebooks/ML_AQI_Complete_Project.ipynb
notebooks/ML_AQI_Results.ipynb
```

Both notebooks have been executed successfully in the verified environment.

---

## Step 9 — Launch Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard provides interactive views of:

- cities,
- forecast horizons,
- models,
- model predictions,
- experiment tables,
- and saved research figures.

---

# 9. Main Chronological Results

## H1 — Next-Day Forecast

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Quantile GBM | **16.73** | 26.16 | 0.915 |
| Gradient Boosting | 17.19 | 26.20 | 0.914 |
| Ridge | 17.47 | 27.24 | 0.908 |
| NumPy Ridge | 17.47 | 27.24 | 0.908 |
| XGBoost | 17.50 | 26.43 | 0.913 |
| Random Forest | 17.51 | 26.74 | 0.911 |
| LSTM | 17.76 | 26.94 | 0.910 |
| PyTorch MLP | 19.92 | 30.28 | 0.886 |
| NumPy MLP | 20.67 | 28.86 | 0.896 |
| Persistence | 21.06 | 34.22 | 0.854 |
| Seasonal Naive | 47.88 | 68.59 | 0.414 |

The strongest H1 model is Quantile GBM.

Its improvement over Persistence is approximately:

$$
20.6\%
$$

The LSTM improves on Persistence by approximately:

$$
15.7\%
$$

---

## H3 — Three-Day Forecast

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge / NumPy Ridge | **29.95** | 46.17 | 0.729 |
| Quantile GBM | 29.99 | **45.81** | **0.734** |
| LSTM | 30.99 | 47.52 | 0.713 |
| Random Forest | 32.04 | 48.11 | 0.706 |
| XGBoost | 32.88 | 47.80 | 0.710 |
| PyTorch MLP | 33.33 | 51.56 | 0.663 |
| Gradient Boosting | 34.29 | 49.36 | 0.691 |
| Persistence | 34.56 | 55.01 | 0.616 |
| NumPy MLP | 35.68 | 52.38 | 0.652 |
| Seasonal Naive | 47.72 | 69.16 | 0.393 |

Ridge improves on Persistence by approximately:

$$
13.3\%
$$

The LSTM improves on Persistence by approximately:

$$
10.3\%
$$

---

# 10. Distribution-Shift Evaluation

The final evaluation includes 17 scenarios per horizon:

```text
Random_Reference
Chronological_Full
Year_2019
Year_2020
Season_Winter
Season_Summer
Season_Monsoon
Season_PostMonsoon
Event_Diwali
Event_StubbleBurning
Event_Lockdown
Event_WinterInversion
City_Bengaluru
City_Chennai
City_Delhi
City_Hyderabad
City_Lucknow
```

The results demonstrate that forecasting difficulty varies substantially across cities, pollution regimes, and horizons.

In particular, H3 forecasting is significantly more difficult than H1 forecasting, and some nonlinear models that perform well at H1 do not consistently outperform Persistence across all H3 city regimes.

---

# 11. Conformal Prediction

The project evaluates:

- static split conformal prediction,
- rolling adaptive conformal recalibration.

Primary metrics:

```text
PICP
MPIW
Coverage Gap
```

where the nominal prediction-interval coverage is:

$$
90\%
$$

Rolling recalibration is particularly important when residual distributions change across seasons or pollution regimes.

---

# 12. Severe Pollution Evaluation

Severe pollution is defined as:

$$
AQI > 400
$$

Metrics include:

- Severe Recall
- Severe Precision
- Severe F1
- MAE by AQI severity bucket

Severity buckets include:

```text
Moderate: 101–200
Poor: 201–300
Very Poor: 301–400
Severe: >400
```

---

# 13. Reproducibility

The project uses:

- chronological train/validation/test separation,
- fixed random seeds,
- training-only statistics for preprocessing,
- explicit target-validity flags,
- saved model checkpoints,
- automated tests,
- reproducible experiment scripts,
- generated result tables,
- and publication figures.

The LSTM sequence pipeline was rerun in the final Python 3.11 environment and reproduced the stored H1/H3 metrics.

Small numerical differences may occur across operating systems, PyTorch backends, or library versions, particularly for stochastic GPU/MPS operations.

---

# 14. Important Output Files

Primary result table:

```text
results/tables/metrics_all_horizons.csv
```

LSTM results:

```text
results/tables/lstm_metrics.csv
```

Ablation tables:

```text
results/tables/ablation_*.csv
```

Spatial evaluation:

```text
results/tables/spatial_held_out_city.csv
```

Secondary dataset:

```text
results/tables/beijing_shift_results.csv
```

Publication figures:

```text
results/figures/
```

---

# 15. Submission Deliverables

The final submission is intended to contain:

- GitHub repository link
- complete source code
- notebooks
- environment requirements
- trained checkpoints
- reproducibility instructions
- reproducibility declaration
- project poster
- experiment tables and figures
- commit hashes and timestamps
- CVPR-format 8-page report

Raw datasets should not be duplicated in the GitHub repository when they can be downloaded from their original sources.

---

# 16. Reproducibility Declaration

The results reported in this project were generated using the scripts and configuration contained in this repository.

The preprocessing, tabular-model evaluation, LSTM evaluation, uncertainty analysis, ablation experiments, secondary Beijing experiment, figure generation, and automated tests can be executed independently using the commands documented above.

The final local verification completed successfully with:

```text
24/24 tests passed
```

and both project notebooks and the Streamlit application were executed successfully in the project-specific Python 3.11 environment.
