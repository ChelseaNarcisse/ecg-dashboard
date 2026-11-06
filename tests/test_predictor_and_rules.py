"""Tests for the prediction layer, the rule-based fallback and the saved models."""

import numpy as np
import pandas as pd
import pytest

from cardio_ml.constants import AAMI_CLASSES
from cardio_ml.features import engineer_features
from cardio_ml.predictor import ModelBundle, full_proba, load_registry, predict_batch, risk_tier, summarize
from cardio_ml.rules import RuleBasedClassifier

from conftest import MODEL_DIR, MockModel, real_models_available


# ---------------- risk tiers ----------------
@pytest.mark.parametrize("score,tier", [(0.0, "low"), (0.29, "low"), (0.3, "moderate"),
                                        (0.69, "moderate"), (0.7, "high"), (1.0, "high")])
def test_risk_tier_boundaries(score, tier):
    assert risk_tier(score) == tier


# ---------------- probability alignment ----------------
def test_full_proba_fills_missing_classes_with_zero(sample_df):
    X = engineer_features(sample_df)
    p = full_proba(MockModel(), X, AAMI_CLASSES)
    assert p.shape == (len(X), 5)
    assert np.allclose(p[:, AAMI_CLASSES.index("Q")], 0)   # mock model has no Q class
    assert np.allclose(p.sum(axis=1), 1)


# ---------------- predict_batch / summarize ----------------
def test_predict_batch_output_contract(sample_df):
    bundle = ModelBundle(name="mock", estimator=MockModel())
    res = predict_batch(bundle, sample_df)
    assert len(res) == len(sample_df)
    expected = {"index", "record", "predicted_label", "confidence", "risk_score", "risk_tier",
                "ventricular_alert", *[f"p_{c}" for c in AAMI_CLASSES]}
    assert expected <= set(res.columns)
    assert np.allclose(res["risk_score"], 1 - res["p_N"], atol=1e-3)
    assert set(res["predicted_label"]) <= set(AAMI_CLASSES)


def test_summarize_counts_add_up(sample_df):
    res = predict_batch(ModelBundle(name="mock", estimator=MockModel()), sample_df)
    s = summarize(res)
    assert s["n_beats"] == len(sample_df)
    assert sum(s["class_counts"].values()) == len(sample_df)
    assert sum(s["risk_tier_counts"].values()) == len(sample_df)
    assert 0 <= s["abnormal_burden_pct"] <= 100


# ---------------- rule-based fallback ----------------
def _feats(qrs_ms, rr_norm):
    return pd.DataFrame({"0_qrs_ms": qrs_ms, "rr_norm_local": rr_norm})


def test_rules_wide_and_early_is_veb():
    assert RuleBasedClassifier().predict(_feats([150], [0.6]))[0] == "VEB"


def test_rules_narrow_and_early_is_sveb():
    assert RuleBasedClassifier().predict(_feats([80], [0.7]))[0] == "SVEB"


def test_rules_on_time_beat_is_normal():
    assert RuleBasedClassifier().predict(_feats([80, 150], [1.0, 1.0])).tolist() == ["N", "N"]


def test_rules_proba_is_valid_distribution():
    p = RuleBasedClassifier().predict_proba(_feats([80, 150, 90], [1.0, 0.5, 0.6]))
    assert np.allclose(p.sum(axis=1), 1)
    assert (p.argmax(axis=1) == [0, 2, 1]).all()


# ---------------- real serialized models (integration) ----------------
@pytest.mark.slow
@pytest.mark.skipif(not real_models_available(), reason="trained models not present")
def test_real_models_load_and_predict(sample_df):
    registry = load_registry(MODEL_DIR)
    assert "rules" in registry and len(registry) >= 2
    for name, bundle in registry.items():
        res = predict_batch(bundle, sample_df)
        assert len(res) == len(sample_df), name
        assert res["risk_score"].between(0, 1).all(), name


@pytest.mark.slow
@pytest.mark.skipif(not real_models_available(), reason="trained models not present")
def test_best_model_detects_ventricular_beats(sample_df):
    """Record 208 (unseen patient) contains many PVCs; the model must find most of them."""
    bundle = load_registry(MODEL_DIR)["hgb"]
    res = predict_batch(bundle, sample_df)
    truth = sample_df["type"].to_numpy()
    veb = truth == "VEB"
    assert veb.sum() > 0
    recall = (res["predicted_label"].to_numpy()[veb] == "VEB").mean()
    assert recall >= 0.7
