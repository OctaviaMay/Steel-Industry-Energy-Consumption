
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from api import service
from api.schemas import (
    EvaluationSummary, ModelInfo, ModelName, PredictRequest,
    PredictResponse, SeriesResponse,
)


@asynccontextmanager
async def lifespan(app):
    # Load models and test predictions once at startup, so requests are fast.
    for name in ModelName:
        service.get_test_predictions(name.value)
    yield


app = FastAPI(
    title="Steel Plant Electricity Forecast API",
    description=(
        "Predicts a steel plant's electricity use (kWh) for the next 15 minutes, "
        "using only information available before that period starts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def home():
    return {'message':'FastAPI is running!'}

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


@app.get("/models", response_model=list[ModelInfo], tags=["system"])
def list_models():
    return [{"name": name, "features": service.FEATURES} for name in ModelName]


@app.post("/predict", response_model=PredictResponse, tags=["forecast"])
def predict(req: PredictRequest):
    features = service.build_features(
        timestamp=req.timestamp,
        load_type=req.load_type.value,
        recent_usage=req.recent_usage_kwh,
        lagging_reactive_power=req.lagging_reactive_power_kvarh,
        lagging_power_factor=req.lagging_power_factor,
    )
    prediction = service.predict_one(req.model.value, features)
    return {
        "predicted_usage_kwh": round(max(prediction, 0.0), 3),  # usage can't be negative
        "model": req.model,
        "naive_baseline_kwh": req.recent_usage_kwh[-1],
        "features": {k: round(float(v), 6) for k, v in features.items()},
    }


@app.get("/evaluation/summary", response_model=EvaluationSummary, tags=["evaluation"])
def evaluation_summary(model: ModelName = ModelName.random_forest):
    return service.evaluation_summary(model.value)


@app.get("/evaluation/series", response_model=SeriesResponse, tags=["evaluation"])
def evaluation_series(
    start: datetime = Query(examples=["2018-11-12T00:00:00"]),
    days: int = Query(3, ge=1, le=14),
    model: ModelName = ModelName.random_forest,
):
    first, last = service.get_test_period()
    window, points = service.evaluation_series(model.value, start, days)
    if not points:
        raise HTTPException(
            status_code=404,
            detail=f"No test data in that range. Test period: {first:%Y-%m-%d} to {last:%Y-%m-%d}.",
        )
    metrics = service.compute_metrics(window)
    if np.isnan(metrics["r2"]):
        metrics["r2"] = 0.0
    return {"model": model, "metrics": metrics, "points": points}