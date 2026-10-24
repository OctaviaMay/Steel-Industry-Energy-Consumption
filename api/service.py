from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "random_forest": ROOT / "models" / "randomforest_regression_pipeline.pkl",
    "xgboost": ROOT / "models" / "xgboost_pipeline.pkl",
}
SPLIT_DIR = ROOT / "data" / "split"

LOAD_TYPE_MAP = {"Light_Load": 1, "Medium_Load": 2, "Maximum_Load": 3}

# get columns in same order as used in model training
FEATURES = [
    "nsm_sin", "nsm_cos", "nsm_sin_loadtype", "nsm_cos_loadtype", "is_weekend",
    "electrical_load", "usage_lag_1", "usage_lag_4",
    "usage_rolling_mean_4", "usage_rolling_std_4",
]


@lru_cache
def get_model(name):
    path = MODEL_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def build_features(timestamp, load_type, recent_usage, lagging_reactive_power, lagging_power_factor):
    # turn user's input raw data into the model's feature row
    ts = pd.Timestamp(timestamp)
    nsm = ts.hour * 3600 + ts.minute * 60
    load = LOAD_TYPE_MAP[load_type]
    usage = np.asarray(recent_usage, dtype=float)

    nsm_sin = np.sin(2 * np.pi * nsm / 86400)
    nsm_cos = np.cos(2 * np.pi * nsm / 86400)
    return {
        "nsm_sin": nsm_sin,
        "nsm_cos": nsm_cos,
        "nsm_sin_loadtype": nsm_sin * load,
        "nsm_cos_loadtype": nsm_cos * load,
        "is_weekend": int(ts.dayofweek >= 5),
        "electrical_load": lagging_reactive_power * (100 - lagging_power_factor),
        "usage_lag_1": usage[-1],
        "usage_lag_4": usage[0],
        "usage_rolling_mean_4": usage.mean(),
        "usage_rolling_std_4": usage.std(ddof=1),
    }


def predict_one(model_name, features):
    row = pd.DataFrame([features], columns=FEATURES)
    return float(get_model(model_name).predict(row)[0])


# --------------------------------------------------------------------------
# Evaluation on the held-out test period
# --------------------------------------------------------------------------

@lru_cache
def _test_set():
    df = pd.read_csv(SPLIT_DIR / "test.csv", parse_dates=["consumed_date"])
    X = pd.read_csv(SPLIT_DIR / "X_test.csv")[FEATURES]
    return df, X


@lru_cache
def get_test_predictions(model_name):
    """Test set with actual, predicted and naive-baseline columns (cached per model)."""
    df, X = _test_set()
    return pd.DataFrame({
        "timestamp": df["consumed_date"],
        "actual": df["Usage_kWh"],
        "predicted": get_model(model_name).predict(X),
        "naive": df["usage_lag_1"],
        "load_type": df["Load_Type"],
        "day_type": df["WeekStatus"],
    })


def compute_metrics(frame):
    model_mae = mean_absolute_error(frame["actual"], frame["predicted"])
    naive_mae = mean_absolute_error(frame["actual"], frame["naive"])
    return {
        "r2": r2_score(frame["actual"], frame["predicted"]) if len(frame) > 1 else float("nan"),
        "mae_kwh": model_mae,
        "naive_mae_kwh": naive_mae,
        "improvement_over_naive": 1 - model_mae / naive_mae if naive_mae else 0.0,
        "n_readings": len(frame),
    }


def _mae_by(frame, column):
    abs_err = (frame["actual"] - frame["predicted"]).abs()
    return abs_err.groupby(frame[column]).mean().round(4).to_dict()


def evaluation_summary(model_name):
    preds = get_test_predictions(model_name)
    return {
        "model": model_name,
        "test_period_start": preds["timestamp"].min(),
        "test_period_end": preds["timestamp"].max(),
        "overall": compute_metrics(preds),
        "mae_by_load_type": _mae_by(preds, "load_type"),
        "mae_by_day_type": _mae_by(preds, "day_type"),
    }


def evaluation_series(model_name, start, days):
    preds = get_test_predictions(model_name)
    start = pd.Timestamp(start)
    window = preds[(preds["timestamp"] >= start) & (preds["timestamp"] < start + pd.Timedelta(days=days))]
    points = [
        {"timestamp": r.timestamp, "actual_kwh": r.actual, "predicted_kwh": r.predicted, "naive_kwh": r.naive}
        for r in window.itertuples()
    ]
    return window, points


def get_test_period():
    df, _ = _test_set()
    return df["consumed_date"].min(), df["consumed_date"].max()