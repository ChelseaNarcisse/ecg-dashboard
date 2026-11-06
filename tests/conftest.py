"""Shared pytest fixtures.

The API tests use a *mocked* model so they are fast, deterministic and do
not depend on the trained .pkl files (TDD: the contract of the endpoint is
tested independently of model quality).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardio_ml.constants import RAW_FEATURE_COLUMNS
from cardio_ml.predictor import ModelBundle
from cardio_ml.rules import RuleBasedClassifier

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CSV = ROOT / "tests" / "fixtures" / "sample_beats.csv"
MODEL_DIR = ROOT / "backend" / "models"


class MockModel:
    """Deterministic stand-in for a trained classifier.

    Beats that arrive early (rr_norm_local < 0.9) get P(VEB) = 0.8,
    every other beat gets P(N) = 0.9.
    """

    classes_ = np.array(["N", "SVEB", "VEB", "F"])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        early = (X["rr_norm_local"] < 0.9).to_numpy()
        proba = np.tile([0.9, 0.05, 0.03, 0.02], (len(X), 1))
        proba[early] = [0.1, 0.05, 0.8, 0.05]
        return proba


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_CSV)


@pytest.fixture
def one_beat(sample_df) -> dict:
    return {c: float(sample_df.iloc[0][c]) for c in RAW_FEATURE_COLUMNS}


@pytest.fixture
def mock_registry() -> dict[str, ModelBundle]:
    return {
        "mock": ModelBundle(name="mock", estimator=MockModel(), display_name="Mock model",
                            metrics={"inter_patient": {"accuracy": 0.99, "balanced_accuracy": 0.9}}),
        "rules": ModelBundle(name="rules", estimator=RuleBasedClassifier(), display_name="Rules"),
    }


@pytest.fixture
def client(mock_registry, tmp_path, monkeypatch):
    """FastAPI TestClient wired to the mock registry and a temp SQLite file."""
    from fastapi.testclient import TestClient

    import app as app_module
    from db import PredictionLog

    monkeypatch.setattr(app_module, "DEFAULT_MODEL", "mock")
    app_module.app.state.registry = mock_registry
    app_module.app.state.log = PredictionLog(tmp_path / "test.db")
    with TestClient(app_module.app) as c:
        yield c
    del app_module.app.state.registry
    del app_module.app.state.log


def to_payload(df: pd.DataFrame, **extra) -> dict:
    cols = [c for c in ["record"] + RAW_FEATURE_COLUMNS if c in df.columns]
    recs = json.loads(df[cols].assign(**({"record": df["record"].astype(str)}
                                         if "record" in df.columns else {})).to_json(orient="records"))
    return {"records": recs, **extra}


def real_models_available() -> bool:
    return any(MODEL_DIR.glob("model_*.pkl"))
