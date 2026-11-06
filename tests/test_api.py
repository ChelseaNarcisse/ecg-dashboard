"""TDD tests for the FastAPI batch endpoint (mocked model + fixture CSV)."""

import time

import pandas as pd
import pytest

from conftest import MODEL_DIR, real_models_available, to_payload


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "mock" in r.json()["models_loaded"]


def test_schema_lists_required_columns(client):
    body = client.get("/schema").json()
    assert len(body["required_columns"]) == 32
    assert "0_pre-RR" in body["required_columns"]


def test_models_endpoint(client):
    names = [m["name"] for m in client.get("/models").json()["models"]]
    assert {"mock", "rules"} <= set(names)


def test_unknown_model_metrics_is_404(client):
    assert client.get("/models/does-not-exist/metrics").status_code == 404


def test_predict_batch_happy_path(client, sample_df):
    r = client.post("/predict-batch?model_type=mock", json=to_payload(sample_df, source_name="fixture"))
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "mock"
    assert body["summary"]["n_beats"] == len(sample_df)
    assert len(body["predictions"]) == len(sample_df)
    first = body["predictions"][0]
    assert first["risk_score"] == pytest.approx(1 - first["p_N"], abs=1e-3)
    assert first["record"] == str(sample_df["record"].iloc[0])


def test_predict_batch_uses_default_model(client, sample_df):
    r = client.post("/predict-batch", json=to_payload(sample_df))
    assert r.status_code == 200 and r.json()["model_used"] == "mock"


def test_predictions_follow_mock_logic(client, sample_df):
    """Mock model labels early beats as VEB - check the API returns exactly that."""
    from cardio_ml.features import engineer_features
    early = (engineer_features(sample_df)["rr_norm_local"] < 0.9).to_numpy()
    preds = client.post("/predict-batch?model_type=mock", json=to_payload(sample_df)).json()["predictions"]
    labels = [p["predicted_label"] for p in preds]
    assert all((lab == "VEB") == e for lab, e in zip(labels, early))


def test_rules_fallback_endpoint(client, sample_df):
    r = client.post("/predict-batch?model_type=rules", json=to_payload(sample_df))
    assert r.status_code == 200


def test_unknown_model_type_is_400(client, sample_df):
    r = client.post("/predict-batch?model_type=nope", json=to_payload(sample_df))
    assert r.status_code == 400


def test_missing_feature_field_is_422(client, one_beat):
    one_beat.pop("0_qt_interval")
    assert client.post("/predict-batch", json={"records": [one_beat]}).status_code == 422


def test_non_numeric_field_is_422(client, one_beat):
    one_beat["0_rPeak"] = "high"
    assert client.post("/predict-batch", json={"records": [one_beat]}).status_code == 422


def test_empty_batch_is_422(client):
    assert client.post("/predict-batch", json={"records": []}).status_code == 422


def test_physiologic_range_violation_is_422(client, one_beat):
    one_beat["0_pre-RR"] = -10
    r = client.post("/predict-batch", json={"records": [one_beat]})
    assert r.status_code == 422
    assert "physiologic" in str(r.json()["detail"])


def test_invalid_sampling_rate_is_422(client, sample_df):
    r = client.post("/predict-batch", json=to_payload(sample_df, sampling_rate_hz=0))
    assert r.status_code == 422


def test_batch_is_logged_to_history(client, sample_df):
    client.post("/predict-batch?model_type=mock", json=to_payload(sample_df, source_name="fixture.csv"))
    runs = client.get("/history").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["source_name"] == "fixture.csv"
    assert runs[0]["n_beats"] == len(sample_df)


def test_failed_batch_is_not_logged(client, one_beat):
    one_beat["0_pre-RR"] = -10
    client.post("/predict-batch", json={"records": [one_beat]})
    assert client.get("/history").json()["runs"] == []


# ---------------- non-functional: performance (proposal NFR) ----------------
@pytest.mark.slow
@pytest.mark.skipif(not real_models_available(), reason="trained models not present")
def test_batch_of_10000_beats_with_real_model(client, sample_df):
    import app as app_module
    from cardio_ml.predictor import load_registry

    app_module.app.state.registry = load_registry(MODEL_DIR)
    big = pd.concat([sample_df] * 170, ignore_index=True)  # 10,200 beats
    t = time.perf_counter()
    r = client.post("/predict-batch?model_type=hgb", json=to_payload(big))
    elapsed = time.perf_counter() - t
    assert r.status_code == 200
    assert r.json()["summary"]["n_beats"] == len(big)
    assert elapsed < 30, f"batch took {elapsed:.1f}s"


def test_falls_back_to_rules_when_no_ml_model_loaded(client, sample_df, mock_registry):
    """Graceful degradation: with only the rule engine loaded, predictions still work."""
    import app as app_module
    app_module.app.state.registry = {"rules": mock_registry["rules"]}
    r = client.post("/predict-batch", json=to_payload(sample_df))
    assert r.status_code == 200
    assert r.json()["model_used"] == "rules"
