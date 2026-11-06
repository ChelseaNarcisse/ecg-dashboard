"""Tests for the dashboard helpers (frontend/dashboard_utils.py)."""

import pandas as pd
import plotly.graph_objects as go
import pytest

import dashboard_utils as du
from cardio_ml.constants import RAW_FEATURE_COLUMNS


def test_frontend_and_backend_agree_on_required_columns():
    assert du.REQUIRED_COLUMNS == RAW_FEATURE_COLUMNS


def test_check_columns(sample_df):
    assert du.check_columns(sample_df) == []
    assert du.check_columns(sample_df.drop(columns=["0_rPeak"])) == ["0_rPeak"]


def test_build_payload_never_sends_true_label(sample_df):
    body = du.build_payload(sample_df, 360, "x.csv")
    assert len(body["records"]) == len(sample_df)
    assert "type" not in body["records"][0]
    assert body["records"][0]["record"] == str(sample_df["record"].iloc[0])
    assert body["sampling_rate_hz"] == 360


@pytest.fixture
def api_response(client, sample_df):
    return client.post("/predict-batch?model_type=mock", json=du.build_payload(sample_df)).json()


def test_frontend_payload_is_accepted_by_api(api_response, sample_df):
    assert api_response["summary"]["n_beats"] == len(sample_df)


def test_evaluate_against_labels(api_response, sample_df):
    pred = du.predictions_frame(api_response, sample_df)
    ev = du.evaluate_against_labels(pred)
    assert 0 <= ev["accuracy"] <= 1
    assert sum(map(sum, ev["confusion_matrix"])) == len(sample_df)
    assert du.evaluate_against_labels(du.predictions_frame(api_response)) is None


def test_per_record_summary_and_flags(api_response, sample_df):
    pred = du.predictions_frame(api_response, sample_df)
    summary = du.per_record_summary(pred)
    assert summary["beats"].sum() == len(sample_df)
    flagged = du.flagged_beats(pred, 0.5)
    assert (flagged["risk_score"] >= 0.5).all()
    assert flagged["risk_score"].is_monotonic_decreasing


def test_all_figures_build(api_response, sample_df):
    from cardio_ml.features import engineer_features
    pred = du.predictions_frame(api_response, sample_df)
    eda = pd.concat([sample_df[["record", "type"]], engineer_features(sample_df)], axis=1)
    figs = [
        du.class_count_bar({"N": 10, "VEB": 3}, log_scale=True),
        du.feature_box_by_class(eda, "0_qrs_ms"),
        du.rhythm_morphology_scatter(eda),
        du.confusion_heatmap([[5, 1], [0, 3]], ["N", "VEB"]),
        du.model_comparison_bar([{"display_name": "m", "accuracy": .9, "balanced_accuracy": .8, "macro_f1": .7}]),
        du.feature_importance_bar([{"feature": "a", "importance": .1, "std": .01, "engineered": True},
                                   {"feature": "b", "importance": .05, "std": .01, "engineered": False}]),
        du.ablation_bar({"hgb": {"raw_only": {"balanced_accuracy": .5},
                                 "raw_plus_engineered": {"balanced_accuracy": .7}}}),
        du.risk_histogram(pred, 0.7),
        du.predicted_class_bar(pred),
        du.risk_timeline(pred, str(sample_df["record"].iloc[0])),
        du.history_trend([{"created_at": "2026-01-01", "abnormal_pct": 5.0, "model_used": "m", "n_beats": 1}]),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)


def test_class_colors_are_fixed_per_class():
    assert len(set(du.CLASS_COLORS.values())) == len(du.CLASSES)
    assert not set(du.CLASS_COLORS.values()) & set(du.TIER_COLORS.values())


@pytest.mark.slow
def test_local_api_client_runs_backend_in_process(sample_df):
    """One-process mode (Streamlit Community Cloud): no API server running."""
    from conftest import ROOT, real_models_available
    if not real_models_available():
        pytest.skip("trained models not available")
    api = du.LocalApiClient(ROOT / "backend")
    assert api.health()["status"] == "ok"
    assert {m["name"] for m in api.models()["models"]} >= {"hgb"}
    resp = api.predict_batch(du.build_payload(sample_df), "hgb")
    assert resp["summary"]["n_beats"] == len(sample_df)
    with pytest.raises(du.ApiError) as err:
        api.predict_batch(du.build_payload(sample_df), "no-such-model")
    assert err.value.status_code == 400
