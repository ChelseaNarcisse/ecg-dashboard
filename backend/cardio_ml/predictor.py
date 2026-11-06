"""Turn a validated batch of beats into predictions + risk scores.

Kept separate from the FastAPI layer so it can be unit tested without a
web server, and so the API stays a thin "controller".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .constants import AAMI_CLASSES, RISK_TIERS, VENTRICULAR_CLASSES
from .features import MODEL_FEATURE_COLUMNS, engineer_features
from .rules import RuleBasedClassifier
from .schema import validate_beats_dataframe

MODEL_FILE_PREFIX = "model_"


@dataclass
class ModelBundle:
    """A trained model plus everything needed to use and describe it."""

    name: str
    estimator: Any
    feature_names: list[str] = field(default_factory=lambda: list(MODEL_FEATURE_COLUMNS))
    classes: list[str] = field(default_factory=lambda: list(AAMI_CLASSES))
    metrics: dict = field(default_factory=dict)
    display_name: str = ""


def risk_tier(score: float) -> str:
    tier = RISK_TIERS[0][1]
    for threshold, name in RISK_TIERS:
        if score >= threshold:
            tier = name
    return tier


def full_proba(estimator: Any, X: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """predict_proba re-ordered to AAMI_CLASSES (missing classes -> 0)."""
    proba = estimator.predict_proba(X)
    est_classes = [str(c) for c in getattr(estimator, "classes_", classes)]
    out = np.zeros((len(X), len(AAMI_CLASSES)))
    for j, cls in enumerate(est_classes):
        if cls in AAMI_CLASSES:
            out[:, AAMI_CLASSES.index(cls)] = proba[:, j]
    return out


def predict_batch(bundle: ModelBundle, df: pd.DataFrame, sampling_rate_hz: float = 360) -> pd.DataFrame:
    """Validate -> engineer features -> predict. Returns one row per beat."""
    validate_beats_dataframe(df)
    feats = engineer_features(df, sampling_rate_hz=sampling_rate_hz)
    X = feats[bundle.feature_names]
    proba = full_proba(bundle.estimator, X, bundle.classes)

    pred_idx = proba.argmax(axis=1)
    labels = np.array(AAMI_CLASSES)[pred_idx]
    risk = 1.0 - proba[:, AAMI_CLASSES.index("N")]

    result = pd.DataFrame({
        "index": np.arange(len(df)),
        "predicted_label": labels,
        "confidence": proba.max(axis=1),
        "risk_score": np.round(risk, 4),
    })
    result["risk_tier"] = [risk_tier(r) for r in result["risk_score"]]
    result["ventricular_alert"] = result["predicted_label"].isin(VENTRICULAR_CLASSES)
    if "record" in df.columns:
        result.insert(1, "record", df["record"].astype(str).to_numpy())
    for j, cls in enumerate(AAMI_CLASSES):
        result[f"p_{cls}"] = np.round(proba[:, j], 4)
    return result


def summarize(results: pd.DataFrame) -> dict:
    counts = results["predicted_label"].value_counts().reindex(AAMI_CLASSES, fill_value=0)
    tiers = results["risk_tier"].value_counts().reindex(["low", "moderate", "high"], fill_value=0)
    return {
        "n_beats": int(len(results)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "risk_tier_counts": {k: int(v) for k, v in tiers.items()},
        "mean_risk": float(results["risk_score"].mean()),
        "n_high_risk": int(tiers["high"]),
        "n_ventricular_alerts": int(results["ventricular_alert"].sum()),
        "abnormal_burden_pct": float(100.0 * (results["predicted_label"] != "N").mean()),
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_bundle(path: str | Path) -> ModelBundle:
    obj = joblib.load(path)
    if isinstance(obj, ModelBundle):
        return obj
    if isinstance(obj, dict):  # saved by training/train_models.py
        return ModelBundle(
            name=obj["name"],
            estimator=obj["estimator"],
            feature_names=obj["feature_names"],
            classes=obj["classes"],
            metrics=obj.get("metrics", {}),
            display_name=obj.get("display_name", obj["name"]),
        )
    raise TypeError(f"Unrecognised model file format: {type(obj)}")


def load_registry(model_dir: str | Path) -> dict[str, ModelBundle]:
    """Load every models/model_*.pkl plus the rule-based fallback."""
    model_dir = Path(model_dir)
    registry: dict[str, ModelBundle] = {}
    for path in sorted(model_dir.glob(f"{MODEL_FILE_PREFIX}*.pkl")):
        bundle = load_bundle(path)
        registry[bundle.name] = bundle
    registry["rules"] = ModelBundle(
        name="rules",
        estimator=RuleBasedClassifier(),
        display_name="Rule-based triage (fallback)",
        metrics=_load_rules_metrics(model_dir),
    )
    return registry


def _load_rules_metrics(model_dir: Path) -> dict:
    import json
    path = model_dir / "metrics.json"
    if path.exists():
        return json.loads(path.read_text()).get("models", {}).get("rules", {})
    return {}
