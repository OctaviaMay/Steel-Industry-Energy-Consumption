# Steel Industry Energy Consumption — Regression Forecasting

Forecasting electricity usage (kWh) for a steel manufacturing plant from operational and time-based signals, using a leakage-aware feature set and a business-defined KPI framework rather than raw accuracy alone.

## Overview

Manufacturers that can anticipate near-term electricity demand can schedule heavy loads more efficiently, negotiate better tariffs, and catch abnormal consumption early. This project builds a supervised regression pipeline that predicts `Usage_kWh` for a steel plant 15 minutes at a time, using only information that would realistically be available *before* that consumption is measured — an explicit design constraint that shaped every modeling decision below.

The project is structured as a small, reproducible pipeline (`src/`) driven by four sequential notebooks (`notebooks/`), rather than one long analysis script, so each stage — data validation, feature engineering, modeling, and business evaluation — can be run, reviewed, and re-run independently.

## Dataset

**Steel Industry Energy Consumption** ([UCI Machine Learning Repository, dataset #851](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)) — 35,040 records of 15-minute interval readings collected in 2018 from a steel plant (DAEWOO Steel Co., Ltd., Gwangyang, South Korea).

| Column | Description |
|---|---|
| `date` | Timestamp (15-minute resolution) |
| `Usage_kWh` | Electricity consumption — **target variable** |
| `Lagging_Current_Reactive.Power_kVarh` | Lagging reactive power |
| `Leading_Current_Reactive_Power_kVarh` | Leading reactive power |
| `CO2(tCO2)` | Carbon emissions |
| `Lagging_Current_Power_Factor` | Lagging power factor (%) |
| `Leading_Current_Power_Factor` | Leading power factor (%) |
| `NSM` | Number of seconds from midnight |
| `WeekStatus` | Weekday / Weekend |
| `Day_of_week` | Day name |
| `Load_Type` | Light / Medium / Maximum load |

The data has no missing values and no duplicate rows.

## Key EDA findings

- `Usage_kWh` is strongly right-skewed and, on a log scale, distinctly **bimodal** — the plant appears to operate in two different consumption regimes rather than one continuous distribution.
- `CO2(tCO2)` correlates with `Usage_kWh` at **r = 0.988** while carrying only 8 unique values across 35,040 rows — a strong signal that it is derived directly from usage rather than independently measured, so it was **dropped as a leakage source**.
- `Load_Type` and `WeekStatus` both separate consumption levels clearly: maximum-load periods and weekdays consume noticeably more electricity than light-load periods and weekends.
- `Leading_Current_Power_Factor` sits at or near 100% for the large majority of readings, so it carries little discriminative information on its own.

## Feature engineering

Beyond dropping the leaked `CO2` column, the pipeline (`src/features/build_features.py`) engineers:

- **Cyclical time features** — `nsm_sin` / `nsm_cos` (sine/cosine encoding of seconds-since-midnight) so midnight wraps around cleanly, plus their interaction with `Load_Type`.
- **Calendar features** — `is_weekend`.
- **Power-quality features** — `total_reactive`, `reactive_mode`, `pf_loss`, `electrical_load`, `near_unity_pf`, derived from the reactive power and power factor readings.
- **Lag/rolling features** (`src/models/train_model.py` pipeline) — `usage_lag_1`, `usage_lag_4`, and a rolling 4-period mean/std of `Usage_kWh`, added after sorting strictly by timestamp.
- Right-skewed features are log-transformed; the notebooks note that any Yeo-Johnson transform of left-skewed features is deliberately deferred until after the train/test split, to avoid fitting transformation parameters on data the model shouldn't see yet.

### Forecast-safety pass

A second leakage check in the modeling notebook goes further than dropping `CO2`: because several of the raw electrical readings are measured in the *same* 15-minute window as the target, using them directly would make the problem trivially easy (near-perfect R²) but unusable for genuine forecasting. The finalized feature set therefore excludes same-window readings such as the raw reactive-power/power-factor columns and `pf_loss`/`total_reactive`, keeping only cyclical time features, load-type interactions, `is_weekend`, `electrical_load`, and the lag/rolling usage features.

> **Note for reviewers:** `electrical_load` is computed as `Lagging_Current_Reactive.Power_kVarh × (100 − Lagging_Current_Power_Factor)` — i.e. from the same two same-window electrical readings that `pf_loss`/`total_reactive` were excluded for using. It's currently the second most important feature for both finalist models (see below), so it's worth double-checking whether it should be lagged by one period (matching `usage_lag_1`) rather than used at the current timestep, to fully honor the forecast-safety rule the rest of the feature set follows.

## Modeling

Three regressors are compared inside `sklearn` pipelines (`StandardScaler` + `OneHotEncoder`, tuned with 5-fold `GridSearchCV`): Linear Regression, Random Forest, and XGBoost. The split is **chronological** (most recent 30% held out as the test set, ~24.5k train / ~10.5k test rows) rather than random, so the test set represents a genuinely future period instead of time-adjacent neighbors of the training data.

| Model | R² | MAE (kWh) |
|---|---|---|
| Linear Regression | 0.885 | 6.27 |
| Random Forest | **0.926** | 3.95 |
| XGBoost | 0.926 | 4.10 |

Linear Regression is dropped early as a clear underperformer; Random Forest and XGBoost carry forward as finalists. For both, `usage_lag_1` dominates feature importance (~82–87%), followed by `electrical_load` and the cyclical time features — intuitively, the best predictor of usage right now is usage a moment ago.

## Business KPI evaluation

Rather than stopping at R²/MAE, the project defines a small business-style KPI table and checks both finalists against it (`04_model_evaluation.ipynb`):

| Metric | Threshold | Random Forest | XGBoost |
|---|---|---|---|
| R² (forecast-safe features) | ≥ 0.75 | 0.926 — **Pass** | 0.926 — **Pass** |
| MAE improvement vs. naive persistence baseline | ≥ 30% | 27.1% — **Fail** | 24.3% — **Fail** |
| MAE by `Load_Type` ≤ 1.5× overall MAE | all classes | Light_Load passes; Medium/Maximum not confirmed passing | Light_Load passes; Medium/Maximum not confirmed passing |
| MAE gap, weekday vs. weekend | < 5 kWh | 3.06 kWh — **Pass** | 3.25 kWh — **Pass** |

**Random Forest is the better of the two finalists** on MAE, load-type consistency, and the weekday/weekend gap, so it's the recommended model — but it's worth being upfront that it does not clear all four KPI bars: its improvement over a naive "same as last period" baseline falls short of the 30% target, and the per-load-type consistency check has only been confirmed passing for `Light_Load` in the current notebook output. A natural next step is to either revisit the 30% threshold against what's realistic for 15-minute industrial load forecasting, or invest further feature/model effort specifically at closing that gap — and to report the full pass/fail table (not just the passing rows) so the KPI check reads as a genuine gate rather than a summary of the good news.

## Project structure

```
├── data/
│   ├── raw/              # Original UCI CSV
│   ├── interim/          # After basic cleaning (date parsing, etc.)
│   ├── processed/        # After feature engineering
│   └── split/            # Held-out test set (df/X/y)
├── models/               # Serialized finalist pipelines (joblib)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/make_dataset.py       # Load, validate, clean
│   ├── features/build_features.py # Leakage checks, feature engineering
│   ├── models/train_model.py      # Train/tune/save pipelines
│   ├── models/predict_model.py    # Load models, run KPI evaluation
│   └── visualization/visualize.py # Shared plotting helpers
├── requirements.txt
└── pyproject.toml
```

## Getting started

```bash
git clone https://github.com/OctaviaMay/Steel-Industry-Energy-Consumption.git
cd Steel-Industry-Energy-Consumption
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run the notebooks in order — `01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_modeling.ipynb` → `04_model_evaluation.ipynb` — from the `notebooks/` directory. Each stage reads its input from `data/` and writes its output back there (or to `models/`) for the next notebook to pick up.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, joblib, Jupyter.

## Limitations & next steps

- Tighten the `electrical_load` feature (see note above) so every feature in the finalist models is unambiguously available before the target period is measured.
- Report the complete KPI pass/fail table, including the currently-hidden `Medium_Load`/`Maximum_Load` rows and the failing MAE-improvement row, so the evaluation reads as a full gate rather than a curated summary.
- Consider `TimeSeriesSplit` in place of standard k-fold `GridSearchCV`, since the data is sequential.
- The `tests/` directory is currently a placeholder — adding unit tests for the feature-engineering and evaluation functions in `src/` would round out the pipeline.
- LightGBM/CatBoost comparisons and threshold-tuning experiments are natural extensions of the current finalist comparison.

## Author

Octavia — [GitHub](https://github.com/OctaviaMay)
