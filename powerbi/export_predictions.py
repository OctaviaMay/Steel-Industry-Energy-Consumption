
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "split"

test = pd.read_csv(SPLIT / "test.csv")
X_test = pd.read_csv(SPLIT / "X_test.csv")
y_test = pd.read_csv(SPLIT / "y_test.csv")
model = joblib.load(ROOT / "models" / "randomforest_regression_pipeline.pkl")

out = pd.DataFrame({
    "timestamp": test["consumed_date"],
    "Load_Type": test["Load_Type"],
    "WeekStatus": test["WeekStatus"],
    "actual_kWh": y_test["Usage_kWh"].values,
    "predicted_kWh": np.round(model.predict(X_test), 3),
    "baseline_kWh": X_test["usage_lag_1"].values,  # naive persistence: previous 15-min value
})
out["abs_error"] = (out["actual_kWh"] - out["predicted_kWh"]).abs().round(3)
out["baseline_abs_error"] = (out["actual_kWh"] - out["baseline_kWh"]).abs().round(3)

out.to_csv(ROOT / "powerbi" / "steel_predictions.csv", index=False)
print(f"Saved {len(out):,} rows. MAE {out['abs_error'].mean():.3f} vs baseline {out['baseline_abs_error'].mean():.3f}")