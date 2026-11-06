"""FastAPI inference service for ECG arrhythmia classification.

Endpoints
---------
GET  /health                 liveness + which models are loaded
GET  /models                 list of models with headline metrics
GET  /models/{name}/metrics  full evaluation (confusion matrix, importances)
GET  /metrics/summary        dataset, ablation, leakage & external-validation results
GET  /schema                 required input columns for a batch
POST /predict-batch          score a whole batch of beats in one request
GET  /history                recent batch runs (SQLite log)

Run locally:  uvicorn app:app --reload --port 8000   (from backend/)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, create_model

from cardio_ml.constants import (
    AAMI_CLASSES, CLASS_DESCRIPTIONS, MAX_BATCH_SIZE, RAW_FEATURE_COLUMNS,
)
from cardio_ml.predictor import ModelBundle, load_registry, predict_batch, summarize
from cardio_ml.schema import SchemaValidationError
from db import PredictionLog

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "predictions.db"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "hgb")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

# One field per raw feature. The CSV column names ("0_pre-RR") are not
# valid Python identifiers, so each field gets a safe name plus an alias.
def _field_name(col: str) -> str:
    return "f_" + col.replace("-", "_")


BeatFeatures = create_model(
    "BeatFeatures",
    __config__=ConfigDict(populate_by_name=True, extra="ignore"),
    record=(Optional[str], Field(default=None, description="Recording / patient id")),
    **{_field_name(c): (float, Field(..., alias=c)) for c in RAW_FEATURE_COLUMNS},
)


class BatchRequest(BaseModel):
    records: list[BeatFeatures] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)  # type: ignore[valid-type]
    sampling_rate_hz: float = Field(360.0, gt=0, le=2000,
                                    description="Sampling rate the interval features were measured at")
    source_name: Optional[str] = Field(None, max_length=200, description="e.g. uploaded file name")


class BeatPrediction(BaseModel):
    index: int
    record: Optional[str] = None
    predicted_label: str
    confidence: float
    risk_score: float
    risk_tier: Literal["low", "moderate", "high"]
    ventricular_alert: bool
    p_N: float
    p_SVEB: float
    p_VEB: float
    p_F: float
    p_Q: float


class BatchSummary(BaseModel):
    n_beats: int
    class_counts: dict[str, int]
    risk_tier_counts: dict[str, int]
    mean_risk: float
    n_high_risk: int
    n_ventricular_alerts: int
    abnormal_burden_pct: float


class BatchResponse(BaseModel):
    batch_id: str
    model_used: str
    latency_ms: float
    summary: BatchSummary
    predictions: list[BeatPrediction]


# ---------------------------------------------------------------------------
# App state & dependencies (overridable in tests)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are loaded ONCE at start-up, not per request.
    if not hasattr(app.state, "registry"):
        app.state.registry = load_registry(MODEL_DIR)
    if not hasattr(app.state, "log"):
        app.state.log = PredictionLog(DB_PATH)
    yield


app = FastAPI(
    title="ECG Arrhythmia Batch Prediction API",
    description="Classifies heartbeats into AAMI classes (N, SVEB, VEB, F, Q) "
                "and returns a per-beat abnormality risk score.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_registry(request: Request) -> dict[str, ModelBundle]:
    return request.app.state.registry


def get_log(request: Request) -> PredictionLog:
    return request.app.state.log


def resolve_default_model(registry: dict) -> str:
    """Graceful degradation: use DEFAULT_MODEL if loaded, otherwise any
    trained ML model, otherwise the rule-based triage fallback."""
    if DEFAULT_MODEL in registry:
        return DEFAULT_MODEL
    ml_models = [n for n in registry if n != "rules"]
    return ml_models[0] if ml_models else "rules"


def _metrics_file() -> dict:
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health(registry: dict = Depends(get_registry)):
    return {"status": "ok", "models_loaded": sorted(registry),
            "default_model": resolve_default_model(registry)}


@app.get("/schema")
def schema():
    return {
        "required_columns": RAW_FEATURE_COLUMNS,
        "optional_columns": ["record", "type"],
        "classes": AAMI_CLASSES,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "max_batch_size": MAX_BATCH_SIZE,
    }


@app.get("/models")
def list_models(registry: dict = Depends(get_registry)):
    out = []
    for name, bundle in registry.items():
        ip = bundle.metrics.get("inter_patient", {})
        out.append({
            "name": name,
            "display_name": bundle.display_name or name,
            "accuracy": ip.get("accuracy"),
            "balanced_accuracy": ip.get("balanced_accuracy"),
            "macro_f1": ip.get("macro_f1"),
            "abnormal_auc": ip.get("abnormal_auc"),
        })
    return {"default_model": resolve_default_model(registry), "models": out}


@app.get("/models/{name}/metrics")
def model_metrics(name: str, registry: dict = Depends(get_registry)):
    if name not in registry:
        raise HTTPException(status_code=404, detail=f"Unknown model '{name}'")
    return {"name": name, **registry[name].metrics}


@app.get("/metrics/summary")
def metrics_summary():
    m = _metrics_file()
    return {k: m.get(k) for k in ("dataset", "ablation", "random_split",
                                  "external_incart", "best_model", "trained_at")}


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch_endpoint(
    payload: BatchRequest,
    model_type: Optional[str] = Query(None, description="logistic | rf | hgb | xgb | rules "
                                                        "(omit to use the default model)"),
    registry: dict = Depends(get_registry),
    log: PredictionLog = Depends(get_log),
):
    if model_type is None:
        model_type = resolve_default_model(registry)
    if model_type not in registry:
        raise HTTPException(status_code=400,
                            detail=f"Unknown model_type '{model_type}'. Available: {sorted(registry)}")
    start = time.perf_counter()
    df = pd.DataFrame([r.model_dump(by_alias=True) for r in payload.records])
    try:
        results = predict_batch(registry[model_type], df, sampling_rate_hz=payload.sampling_rate_hz)
    except SchemaValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors)
    summary = summarize(results)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    batch_id = uuid.uuid4().hex[:12]
    log.log_batch(batch_id, model_type, summary, latency_ms, payload.source_name)

    if "record" in results.columns:
        results["record"] = results["record"].replace({"None": None})
    return {
        "batch_id": batch_id,
        "model_used": model_type,
        "latency_ms": latency_ms,
        "summary": summary,
        "predictions": results.to_dict(orient="records"),
    }


@app.get("/history")
def history(limit: int = Query(50, ge=1, le=500), log: PredictionLog = Depends(get_log)):
    return {"runs": log.recent(limit)}
