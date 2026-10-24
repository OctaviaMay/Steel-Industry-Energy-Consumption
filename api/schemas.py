
from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class ModelName(str, Enum):
    random_forest = "random_forest"
    xgboost = "xgboost"


class LoadType(str, Enum):
    light = "Light_Load"
    medium = "Medium_Load"
    maximum = "Maximum_Load"


class PredictRequest(BaseModel):
    timestamp: datetime = Field(
        description="Timestamp of the 15-minute reading to forecast, as stamped in the dataset.",
        examples=["2018-11-12T10:00:00"],
    )
    load_type: LoadType = Field(examples=["Medium_Load"])
    recent_usage_kwh: List[float] = Field(
        min_length=4, max_length=4,
        description="The previous four 15-minute usage readings (kWh), oldest first.",
        examples=[[52.1, 49.8, 55.3, 51.0]],
    )
    lagging_reactive_power_kvarh: float = Field(ge=0, examples=[32.4])
    lagging_power_factor: float = Field(ge=0, le=100, examples=[78.5])
    model: ModelName = ModelName.random_forest


class PredictResponse(BaseModel):
    predicted_usage_kwh: float
    model: ModelName
    naive_baseline_kwh: float = Field(description="Last reading, i.e. 'same as the previous 15 minutes'.")
    features: dict


class ModelInfo(BaseModel):
    name: ModelName
    features: List[str]


class Metrics(BaseModel):
    r2: float
    mae_kwh: float
    naive_mae_kwh: float
    improvement_over_naive: float = Field(description="1 - model MAE / naive MAE")
    n_readings: int


class EvaluationSummary(BaseModel):
    model: ModelName
    test_period_start: datetime
    test_period_end: datetime
    overall: Metrics
    mae_by_load_type: dict
    mae_by_day_type: dict


class SeriesPoint(BaseModel):
    timestamp: datetime
    actual_kwh: float
    predicted_kwh: float
    naive_kwh: float


class SeriesResponse(BaseModel):
    model: ModelName
    metrics: Metrics
    points: List[SeriesPoint]