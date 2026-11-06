"""Pure helper functions for the Streamlit dashboard.

Everything here is independent of Streamlit so it can be unit tested
(tests/test_dashboard_utils.py). app.py only handles layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

CLASSES = ["N", "SVEB", "VEB", "F", "Q"]
CLASS_NAMES = {
    "N": "Normal", "SVEB": "Supraventricular ectopic", "VEB": "Ventricular ectopic",
    "F": "Fusion", "Q": "Unclassifiable",
}
# Fixed categorical order (validated palette, slots 1-5). Colour follows the
# class, never its rank, so a class keeps its colour on every chart.
CLASS_COLORS = {"N": "#2a78d6", "SVEB": "#eb6834", "VEB": "#1baf7a", "F": "#eda100", "Q": "#e87ba4"}
# Status colours are reserved for risk tiers only (always shown with a label).
TIER_COLORS = {"low": "#0ca30c", "moderate": "#fab219", "high": "#d03b3b"}
FEATURE_KIND_COLORS = {"engineered": "#2a78d6", "raw": "#eb6834"}
SEQUENTIAL_BLUE = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#52514e"

REQUIRED_COLUMNS: list[str] = [
    f"{lead}_{f}" for lead in ("0", "1") for f in (
        "pre-RR", "post-RR", "pPeak", "tPeak", "rPeak", "sPeak", "qPeak",
        "qrs_interval", "pq_interval", "qt_interval", "st_interval",
        "qrs_morph0", "qrs_morph1", "qrs_morph2", "qrs_morph3", "qrs_morph4")
]


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class ApiClient:
    """Talks to the FastAPI backend over HTTP (local two-process setup / Docker)."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _send(self, method: str, path: str, params: dict | None = None, json: Any = None):
        return requests.request(method, f"{self.base_url}{path}", params=params,
                                json=json, timeout=self.timeout)

    def _get(self, path: str, **params) -> Any:
        r = self._send("GET", path, params=params or None)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        return self._get("/health")

    def models(self) -> dict:
        return self._get("/models")

    def model_metrics(self, name: str) -> dict:
        return self._get(f"/models/{name}/metrics")

    def metrics_summary(self) -> dict:
        return self._get("/metrics/summary")

    def history(self, limit: int = 100) -> list[dict]:
        return self._get("/history", limit=limit)["runs"]

    def predict_batch(self, payload: dict, model_type: str) -> dict:
        r = self._send("POST", "/predict-batch", params={"model_type": model_type}, json=payload)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text
            raise ApiError(r.status_code, detail)
        return r.json()


class LocalApiClient(ApiClient):
    """Runs the same FastAPI backend *inside* the Streamlit process.

    Used for one-process hosting (e.g. Streamlit Community Cloud), where there
    is no separate API server. Requests still go through FastAPI - routing,
    Pydantic validation and the SQLite log all behave exactly as over HTTP -
    they just don't touch the network. Predictions still come from the API code.
    """

    def __init__(self, backend_dir: str | Path):
        import importlib.util
        import sys

        from fastapi.testclient import TestClient

        backend_dir = Path(backend_dir).resolve()
        if str(backend_dir) not in sys.path:
            sys.path.append(str(backend_dir))  # for `cardio_ml` and `db`
        # Load backend/app.py under its own name: frontend/app.py is also "app".
        spec = importlib.util.spec_from_file_location("ecg_backend_app", backend_dir / "app.py")
        module = importlib.util.module_from_spec(spec)
        # Register before running it: Pydantic/FastAPI resolve the module's
        # (postponed) type annotations by looking the module up in sys.modules.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.base_url = "in-process"
        self.timeout = None
        self._client = TestClient(module.app)
        self._client.__enter__()  # runs the app's lifespan: models are loaded once

    def _send(self, method: str, path: str, params: dict | None = None, json: Any = None):
        return self._client.request(method, path, params=params, json=json)


class ApiError(Exception):
    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API error {status_code}: {detail}")


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------

def check_columns(df: pd.DataFrame) -> list[str]:
    """Return the required columns missing from an uploaded CSV."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def build_payload(df: pd.DataFrame, sampling_rate_hz: float = 360.0,
                  source_name: str | None = None) -> dict:
    """Parsed CSV -> JSON body for POST /predict-batch. The true label column
    ('type'), if present, is NOT sent: the API never sees the answer."""
    cols = REQUIRED_COLUMNS + (["record"] if "record" in df.columns else [])
    body = df[cols].copy()
    if "record" in body.columns:
        body["record"] = body["record"].astype(str)
    records = json.loads(body.to_json(orient="records"))
    return {"records": records, "sampling_rate_hz": sampling_rate_hz, "source_name": source_name}


def predictions_frame(response: dict, uploaded: pd.DataFrame | None = None) -> pd.DataFrame:
    pred = pd.DataFrame(response["predictions"])
    if uploaded is not None and "type" in uploaded.columns:
        pred["true_label"] = uploaded["type"].astype(str).to_numpy()
    return pred


def evaluate_against_labels(pred: pd.DataFrame) -> dict | None:
    """If the upload contained true labels, score the batch (FR-005)."""
    if "true_label" not in pred.columns:
        return None
    y, p = pred["true_label"], pred["predicted_label"]
    present = [c for c in CLASSES if c in set(y)]
    recalls = {c: float((p[y == c] == c).mean()) for c in present}
    cm = pd.crosstab(pd.Categorical(y, CLASSES), pd.Categorical(p, CLASSES), dropna=False)
    return {
        "accuracy": float((y == p).mean()),
        "balanced_accuracy": float(np.mean(list(recalls.values()))) if recalls else float("nan"),
        "recall": recalls,
        "confusion_matrix": cm.to_numpy().tolist(),
    }


def per_record_summary(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per recording/patient, sorted by abnormal-beat burden."""
    key = "record" if "record" in pred.columns and pred["record"].notna().any() else None
    g = pred.assign(record=pred["record"] if key else "batch").groupby("record")
    out = pd.DataFrame({
        "beats": g.size(),
        "mean_risk": g["risk_score"].mean().round(3),
        "high_risk_beats": g["risk_tier"].apply(lambda s: int((s == "high").sum())),
        "ventricular_alerts": g["ventricular_alert"].sum().astype(int),
        "abnormal_burden_pct": g["predicted_label"].apply(lambda s: round(100 * (s != "N").mean(), 1)),
    })
    for c in ("SVEB", "VEB", "F"):
        out[c] = g["predicted_label"].apply(lambda s, c=c: int((s == c).sum()))
    return out.sort_values("abnormal_burden_pct", ascending=False).reset_index()


def flagged_beats(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    cols = [c for c in ["index", "record", "true_label", "predicted_label", "risk_score",
                        "risk_tier", "confidence", "ventricular_alert"] if c in pred.columns]
    return (pred.loc[pred["risk_score"] >= threshold, cols]
                .sort_values("risk_score", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)),
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK), bargap=0.25, legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#d6d5d0")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def class_count_bar(counts: dict[str, int], title: str = "Beats per class",
                    log_scale: bool = False) -> go.Figure:
    classes = [c for c in CLASSES if c in counts]
    vals = [counts[c] for c in classes]
    fig = go.Figure(go.Bar(
        x=[f"{c}<br><sub>{CLASS_NAMES[c]}</sub>" for c in classes], y=vals,
        marker_color=[CLASS_COLORS[c] for c in classes], marker_line_width=0,
        text=[f"{v:,}" for v in vals], textposition="outside",
        hovertemplate="%{x}<br>%{y:,} beats<extra></extra>",
    ))
    if log_scale:
        fig.update_yaxes(type="log", range=[0, np.log10(max(max(vals, default=1), 1)) + 0.35],
                         title="beats (log scale)")
    else:
        fig.update_yaxes(title="beats")
    return _style(fig, title)


def feature_box_by_class(df: pd.DataFrame, feature: str, label_col: str = "type") -> go.Figure:
    classes = [c for c in CLASSES if c in set(df[label_col])]
    fig = go.Figure()
    for c in classes:
        fig.add_trace(go.Box(y=df.loc[df[label_col] == c, feature], name=c,
                             marker_color=CLASS_COLORS[c], boxpoints=False, line_width=1.5))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title=feature)
    return _style(fig, f"{feature} by beat class")


def rhythm_morphology_scatter(df: pd.DataFrame, x: str = "rr_norm_local", y: str = "0_qrs_ms",
                              label_col: str = "type", n: int = 4000) -> go.Figure:
    sample = df.sample(min(n, len(df)), random_state=0)
    fig = go.Figure()
    for c in [c for c in CLASSES if c in set(sample[label_col])]:
        s = sample[sample[label_col] == c]
        fig.add_trace(go.Scattergl(x=s[x], y=s[y], mode="markers", name=f"{c} ({CLASS_NAMES[c]})",
                                   marker=dict(color=CLASS_COLORS[c], size=6, opacity=0.55,
                                               line=dict(width=0.5, color="white")),
                                   hovertemplate=f"{c}<br>{x}=%{{x:.2f}}<br>{y}=%{{y:.0f}}<extra></extra>"))
    fig.add_vline(x=0.9, line_dash="dot", line_color="#9a9993",
                  annotation_text="early beat (<0.9)", annotation_position="top left")
    fig.add_hline(y=120, line_dash="dot", line_color="#9a9993",
                  annotation_text="wide QRS (>120 ms)", annotation_position="top right")
    fig.update_xaxes(title=f"{x} (RR ÷ patient's recent rhythm)", range=[0.2, 1.8])
    fig.update_yaxes(title=f"{y}", range=[0, 300])
    return _style(fig, "Engineered features separate beat types: prematurity vs QRS width", 440)


def confusion_heatmap(cm: list[list[int]], labels: list[str], normalize: bool = True,
                      title: str = "Confusion matrix") -> go.Figure:
    arr = np.asarray(cm, dtype=float)
    keep = [i for i in range(len(labels)) if arr[i].sum() > 0 or arr[:, i].sum() > 0]
    arr = arr[np.ix_(keep, keep)]
    labs = [labels[i] for i in keep]
    shown = arr / np.clip(arr.sum(axis=1, keepdims=True), 1, None) if normalize else arr
    text = [[(f"{shown[i, j]:.0%}<br>({int(arr[i, j]):,})" if normalize else f"{int(arr[i, j]):,}")
             for j in range(len(labs))] for i in range(len(labs))]
    fig = go.Figure(go.Heatmap(
        z=shown, x=labs, y=labs, text=text, texttemplate="%{text}", colorscale=SEQUENTIAL_BLUE,
        zmin=0, zmax=1 if normalize else None, xgap=2, ygap=2, showscale=False,
        hovertemplate="true %{y} → predicted %{x}<br>%{text}<extra></extra>",
    ))
    fig.update_yaxes(title="true class", autorange="reversed")
    fig.update_xaxes(title="predicted class", side="bottom")
    return _style(fig, title + (" (row-normalised = recall)" if normalize else ""), 400)


def model_comparison_bar(models: list[dict], metrics=("accuracy", "balanced_accuracy", "macro_f1")) -> go.Figure:
    names = [m["display_name"] for m in models]
    fig = go.Figure()
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(name=metric.replace("_", " "), x=names, y=[m.get(metric) for m in models],
                             marker_color=color, marker_line_width=0,
                             hovertemplate="%{x}<br>" + metric + " = %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1], title="score (unseen patients, DS2)")
    return _style(fig, "Model comparison on the inter-patient test set")


def feature_importance_bar(importances: list[dict], top: int = 15) -> go.Figure:
    imp = pd.DataFrame(importances).head(top).iloc[::-1]
    kind = np.where(imp["engineered"], "engineered", "raw")
    fig = go.Figure()
    for k in ("engineered", "raw"):
        s = imp[kind == k]
        fig.add_trace(go.Bar(x=s["importance"], y=s["feature"], orientation="h", name=f"{k} feature",
                             marker_color=FEATURE_KIND_COLORS[k], marker_line_width=0,
                             error_x=dict(type="data", array=s["std"], color="#9a9993", thickness=1),
                             hovertemplate="%{y}<br>drop in balanced accuracy: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="permutation importance (drop in balanced accuracy)")
    fig.update_yaxes(categoryorder="array", categoryarray=imp["feature"].tolist())
    return _style(fig, f"Top {top} features", 480)


def ablation_bar(ablation: dict, metric: str = "balanced_accuracy", names: dict | None = None) -> go.Figure:
    names = names or {}
    models = list(ablation)
    fig = go.Figure()
    for key, label, color in [("raw_only", "raw features only", "#eb6834"),
                              ("raw_plus_engineered", "raw + engineered", "#2a78d6")]:
        vals = [ablation[m][key][metric] for m in models]
        fig.add_trace(go.Bar(name=label, x=[names.get(m, m) for m in models], y=vals,
                             marker_color=color, marker_line_width=0, text=[f"{v:.3f}" for v in vals],
                             textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1], title=metric.replace("_", " "))
    return _style(fig, "Feature-engineering ablation (unseen patients)")


def risk_histogram(pred: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=pred["risk_score"], xbins=dict(start=0, end=1.0001, size=0.05),
                                 marker_color="#2a78d6", marker_line=dict(width=2, color="white"),
                                 hovertemplate="risk %{x}<br>%{y:,} beats<extra></extra>"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#d03b3b",
                  annotation_text=f"flag threshold {threshold:.2f}", annotation_position="top")
    fig.update_xaxes(title="risk score = 1 − P(normal)", range=[0, 1])
    fig.update_yaxes(title="beats", type="log")
    return _style(fig, "Risk-score distribution (log count)")


def predicted_class_bar(pred: pd.DataFrame) -> go.Figure:
    counts = pred["predicted_label"].value_counts().reindex(CLASSES, fill_value=0)
    return class_count_bar({c: int(v) for c, v in counts.items() if v > 0 or c != "Q"},
                           "Predicted beat classes", log_scale=True)


def risk_timeline(pred: pd.DataFrame, record: str | None = None, max_points: int = 6000) -> go.Figure:
    """Risk of each beat in time order - a 'rhythm strip' for one recording."""
    df = pred if record is None else pred[pred["record"] == record]
    df = df.head(max_points)
    fig = go.Figure()
    for tier in ("low", "moderate", "high"):
        s = df[df["risk_tier"] == tier]
        fig.add_trace(go.Scattergl(x=s["index"], y=s["risk_score"], mode="markers", name=f"{tier} risk",
                                   marker=dict(color=TIER_COLORS[tier], size=5, opacity=0.7),
                                   customdata=s[["predicted_label"]],
                                   hovertemplate="beat %{x}<br>risk %{y:.2f}<br>%{customdata[0]}<extra></extra>"))
    fig.update_xaxes(title="beat number (time order)")
    fig.update_yaxes(title="risk score", range=[-0.02, 1.02])
    return _style(fig, f"Beat-by-beat risk{' - record ' + str(record) if record else ''}", 340)


def history_trend(runs: list[dict]) -> go.Figure:
    h = pd.DataFrame(runs).sort_values("created_at")
    h["run"] = pd.to_datetime(h["created_at"]).dt.strftime("%b %d %H:%M:%S") + "<br>" + h["model_used"]
    fig = go.Figure(go.Scatter(x=h["run"], y=h["abnormal_pct"], mode="lines+markers",
                               line=dict(width=2, color="#2a78d6"), marker=dict(size=8),
                               customdata=h[["model_used", "n_beats"]],
                               hovertemplate="%{x}<br>abnormal burden %{y:.1f}%<br>"
                                             "model %{customdata[0]}, %{customdata[1]:,} beats<extra></extra>"))
    fig.update_xaxes(type="category", title="batch run")
    fig.update_yaxes(title="abnormal-beat burden (%)", rangemode="tozero")
    return _style(fig, "Abnormal-beat burden across batch runs")
