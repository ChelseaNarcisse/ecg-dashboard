"""Streamlit dashboard: ECG Arrhythmia Early-Warning Dashboard.

All predictions come from the FastAPI backend (no model code in the dashboard).
Run:  streamlit run app.py

* API_URL set (Docker, or a separately started uvicorn): the dashboard calls
  that API over HTTP.
* API_URL not set (Streamlit Community Cloud, or just running this file): the
  same FastAPI app in ../backend is run inside this process - one command,
  no second server needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_utils as du

API_URL = os.getenv("API_URL")  # unset -> run the backend in-process
HERE = Path(__file__).resolve().parent
SAMPLE_DIR = HERE / "sample_data"
BACKEND_DIR = HERE.parent / "backend"
API_LABEL = API_URL or "in-process (backend/app.py)"

st.set_page_config(page_title="ECG Arrhythmia Early Warning", page_icon="🫀", layout="wide")


@st.cache_resource(show_spinner="Loading models…")
def get_api() -> du.ApiClient:
    # Cached: the backend (and its models) is loaded once per server, not per visitor.
    return du.ApiClient(API_URL) if API_URL else du.LocalApiClient(BACKEND_DIR)


api = get_api()


# ---------- small compatibility helpers (Streamlit's width API changed over versions) ----------
def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


@st.cache_data(show_spinner=False)
def load_eda_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / "eda_sample.csv")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_static():
    return api.health(), api.models(), api.metrics_summary()


# ---------------------------------------------------------------- header
st.title("🫀 ECG Arrhythmia Early-Warning Dashboard")
st.caption("Machine-learning classification of heartbeats into AAMI classes "
           "(N, SVEB, VEB, F, Q) from wearable-style two-lead ECG features. "
           "Academic prototype - not for clinical use.")

try:
    health, models_info, summary = fetch_static()
except Exception as exc:  # backend down -> tell the user how to fix it
    hint = ("Start the backend (`uvicorn app:app --port 8000` in backend/) and refresh."
            if API_URL else "Check that backend/ and its models/ folder are next to frontend/.")
    st.error(f"Cannot reach the prediction API ({API_LABEL}). {hint}\n\n{exc}")
    st.stop()

model_list = models_info["models"]
model_names = {m["name"]: m["display_name"] for m in model_list}
dataset = summary.get("dataset") or {}

with st.sidebar:
    st.header("Settings")
    st.success(f"API online · {len(model_list)} models loaded")
    default = models_info.get("default_model", model_list[0]["name"])
    model_type = st.selectbox("Prediction model", list(model_names), format_func=model_names.get,
                              index=list(model_names).index(default) if default in model_names else 0)
    threshold = st.slider("High-risk flag threshold (risk score)", 0.1, 0.95, 0.7, 0.05)
    st.caption(f"API: {API_LABEL}")

tab_eda, tab_fe, tab_perf, tab_batch, tab_hist = st.tabs(
    ["📊 Data & EDA", "🛠️ Feature engineering", "🎯 Model performance", "📤 Batch prediction", "🕑 History"])

# ================================================================= 1. EDA
with tab_eda:
    eda = load_eda_sample()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beats (MIT-BIH)", f"{dataset.get('n_beats', 0):,}")
    c2.metric("Patients / records", dataset.get("n_records", "-"))
    c3.metric("Raw → model features",
              f"{dataset.get('n_raw_features', 32)} → {dataset.get('n_model_features', '-')}")
    c4.metric("Normal beats", f"{100 * dataset.get('class_counts', {}).get('N', 0) / max(dataset.get('n_beats', 1), 1):.1f}%")

    left, right = st.columns(2)
    with left:
        show_chart(du.class_count_bar(dataset.get("class_counts", {}),
                                      "Class distribution (log scale) - heavily imbalanced", log_scale=True))
    with right:
        split = pd.DataFrame({"train (DS1)": dataset.get("train_class_counts", {}),
                              "test (DS2)": dataset.get("test_class_counts", {})})
        st.markdown("**Inter-patient split** - the model is tested on 22 patients it never saw.")
        show_table(split)
        st.caption(f"{dataset.get('n_q_beats_excluded', 15)} 'Q' beats were excluded: too few to learn from. "
                   f"Redundant duplicate columns dropped: {', '.join(dataset.get('redundant_columns_dropped', []))}.")

    st.subheader("Explore a feature by class")
    st.caption("Stratified sample (≤2,500 beats per class) from the training data.")
    feat_cols = [c for c in eda.columns if c not in ("record", "type")]
    default_feat = feat_cols.index("0_qrs_ms") if "0_qrs_ms" in feat_cols else 0
    feature = st.selectbox("Feature", feat_cols, index=default_feat)
    show_chart(du.feature_box_by_class(eda, feature))

# ================================================================= 2. Feature engineering
with tab_fe:
    st.markdown(
        "Raw features are measured in **samples** and absolute amplitudes. The pipeline converts them "
        "into clinically meaningful, *patient-relative* features - the way a cardiologist reads a strip.")
    groups = pd.DataFrame([
        ["Unit conversion", "samples → milliseconds (360 Hz MIT-BIH, 257 Hz INCART)", "0_qrs_ms, 0_qt_ms, pre_rr_ms"],
        ["Rhythm / prematurity", "RR relative to the patient's own recent rhythm (causal 10-beat window)",
         "rr_ratio, rr_norm_local, rr_norm_record, rr_diff_ms, heart_rate_bpm"],
        ["Intervals", "QRS width, Bazett-corrected QT, wide-QRS flag (>120 ms)", "qtc_bazett_ms, wide_qrs_flag"],
        ["Morphology", "QRS amplitude, R/S and T/R ratios, QRS energy & slope, cross-lead R difference",
         "0_qrs_amplitude, 0_r_s_ratio, 0_qrs_energy, lead_r_diff"],
        ["Cleaning", "4 duplicated columns removed (identical in 100% of beats)", "1_pre-RR, 1_post-RR, *_qrs_morph0"],
    ], columns=["Group", "What it captures", "Examples"])
    show_table(groups, hide_index=True)
    show_chart(du.rhythm_morphology_scatter(eda))
    st.caption("Ventricular beats (VEB) sit top-left: they arrive early **and** have a wide QRS. "
               "SVEB beats are early but narrow; normal beats cluster around rr_norm_local = 1. "
               "These two engineered axes are exactly what the rule-based fallback uses.")
    abl = summary.get("ablation") or {}
    if abl:
        show_chart(du.ablation_bar(abl, names=model_names))
        hgb = abl.get("hgb")
        if hgb:
            gain = hgb["raw_plus_engineered"]["balanced_accuracy"] - hgb["raw_only"]["balanced_accuracy"]
            st.info(f"For gradient boosting, engineered features raise balanced accuracy on unseen patients by "
                    f"**{gain * 100:+.1f} points**. The linear model does *not* benefit - the ratios are "
                    f"non-linear, heavy-tailed signals that tree models exploit better.")

# ================================================================= 3. Model performance
with tab_perf:
    ml_models = [m for m in model_list if m.get("accuracy") is not None]
    show_chart(du.model_comparison_bar(ml_models))
    chosen = st.selectbox("Model details", [m["name"] for m in ml_models],
                          format_func=model_names.get,
                          index=[m["name"] for m in ml_models].index(model_type)
                          if model_type in [m["name"] for m in ml_models] else 0)
    detail = api.model_metrics(chosen)
    ip = detail["inter_patient"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{ip['accuracy']:.3f}")
    k2.metric("Balanced accuracy", f"{ip['balanced_accuracy']:.3f}")
    k3.metric("Macro F1", f"{ip['macro_f1']:.3f}")
    k4.metric("Abnormal-beat AUC", f"{ip.get('abnormal_auc', float('nan')):.3f}")
    a, b = st.columns(2)
    with a:
        show_chart(du.confusion_heatmap(ip["confusion_matrix"], ip["labels"],
                                        title=f"{model_names.get(chosen)} - unseen patients"))
    with b:
        per = pd.DataFrame(ip["per_class"]).T.loc[["N", "SVEB", "VEB", "F"]]
        per.columns = ["precision", "recall", "F1", "support"]
        st.markdown("**Per-class performance**")
        show_table(per.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "F1": "{:.3f}", "support": "{:,.0f}"}))
    if detail.get("feature_importance"):
        show_chart(du.feature_importance_bar(detail["feature_importance"]))

    st.subheader("Why the evaluation design matters")
    rs, ext = summary.get("random_split") or {}, summary.get("external_incart") or {}
    rows = []
    for name in ("logistic", "hgb"):
        if name in rs:
            rows.append({"model": model_names.get(name, name), "evaluation": "Random split (same patients)",
                         **{k: rs[name][k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}})
        if name in (summary.get("ablation") or {}):
            m = next(x for x in model_list if x["name"] == name)
            rows.append({"model": model_names.get(name, name), "evaluation": "Inter-patient (unseen patients)",
                         **{k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}})
        if name in ext:
            rows.append({"model": model_names.get(name, name), "evaluation": "External: INCART (other hospital)",
                         **{k: ext[name][k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}})
    if rows:
        show_table(pd.DataFrame(rows).style.format({"accuracy": "{:.3f}", "balanced_accuracy": "{:.3f}",
                                                   "macro_f1": "{:.3f}"}), hide_index=True)
        st.caption("A random split puts beats from the same patient in both train and test, which inflates "
                   "scores. The inter-patient and external results are the honest estimate for a new wearer.")

# ================================================================= 4. Batch prediction
with tab_batch:
    st.markdown("Upload a CSV of beats (same 32 feature columns as the Kaggle file; optional `record` and `type`). "
                "The whole file is sent to **`POST /predict-batch`** in one request.")
    samples = sorted(p.name for p in SAMPLE_DIR.glob("*.csv") if p.name != "eda_sample.csv")
    src_col, rate_col = st.columns([3, 1])
    with src_col:
        upload = st.file_uploader("Upload beats CSV", type="csv")
        sample_choice = st.selectbox("…or use a demo file (unseen DS2 patients)", ["-"] + samples)
    with rate_col:
        rate = st.number_input("Sampling rate (Hz)", 50, 2000, 360,
                               help="MIT-BIH = 360 Hz, INCART = 257 Hz. Needed to convert samples to ms.")

    df_up, source = None, None
    if upload is not None:
        df_up, source = pd.read_csv(upload), upload.name
    elif sample_choice != "-":
        df_up, source = pd.read_csv(SAMPLE_DIR / sample_choice), sample_choice

    if df_up is not None:
        missing = du.check_columns(df_up)
        st.write(f"**{source}** - {len(df_up):,} beats, {df_up.shape[1]} columns")
        if missing:
            st.error(f"Missing required columns: {missing}")
        elif st.button(f"Run batch prediction with {model_names[model_type]}", type="primary"):
            with st.spinner(f"Scoring {len(df_up):,} beats via the API…"):
                try:
                    resp = api.predict_batch(du.build_payload(df_up, rate, source), model_type)
                    st.session_state["batch"] = (resp, du.predictions_frame(resp, df_up))
                except du.ApiError as err:
                    st.error(f"The API rejected the batch ({err.status_code}): {err.detail}")
                except Exception as err:
                    st.error(f"Request failed: {err}")

    if "batch" in st.session_state:
        resp, pred = st.session_state["batch"]
        s = resp["summary"]
        st.divider()
        st.subheader(f"Results · batch {resp['batch_id']} · {model_names.get(resp['model_used'], resp['model_used'])}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Beats scored", f"{s['n_beats']:,}")
        m2.metric("Abnormal burden", f"{s['abnormal_burden_pct']:.1f}%")
        m3.metric(f"Flagged (risk ≥ {threshold:.2f})", f"{int((pred.risk_score >= threshold).sum()):,}")
        m4.metric("⚠️ Ventricular alerts", f"{s['n_ventricular_alerts']:,}")
        m5.metric("API latency", f"{resp['latency_ms']:,.0f} ms")

        h1, h2 = st.columns(2)
        with h1:
            show_chart(du.risk_histogram(pred, threshold))
        with h2:
            show_chart(du.predicted_class_bar(pred))

        records = pred["record"].dropna().unique().tolist() if "record" in pred.columns else []
        rec = st.selectbox("Rhythm strip for record", records) if len(records) > 1 else (records[0] if records else None)
        show_chart(du.risk_timeline(pred, rec))

        st.markdown("**Per-patient summary** (sorted by abnormal burden)")
        show_table(du.per_record_summary(pred), hide_index=True)

        st.markdown(f"**Flagged high-risk beats** (risk ≥ {threshold:.2f}) - click a column to sort")
        flagged = du.flagged_beats(pred, threshold)
        show_table(flagged, hide_index=True, height=320)
        st.download_button("Download all predictions (CSV)", pred.to_csv(index=False),
                           file_name=f"predictions_{resp['batch_id']}.csv", mime="text/csv")

        ev = du.evaluate_against_labels(pred)
        if ev:
            st.subheader("Accuracy against the labels in this file")
            e1, e2 = st.columns([1, 2])
            with e1:
                st.metric("Accuracy", f"{ev['accuracy']:.3f}")
                st.metric("Balanced accuracy", f"{ev['balanced_accuracy']:.3f}")
                show_table(pd.Series(ev["recall"], name="recall").to_frame().style.format("{:.3f}"))
            with e2:
                show_chart(du.confusion_heatmap(ev["confusion_matrix"], du.CLASSES, title="This batch"))

# ================================================================= 5. History
with tab_hist:
    runs = api.history(100)
    if not runs:
        st.info("No batch runs logged yet - run a batch prediction first.")
    else:
        show_chart(du.history_trend(runs))
        show_table(pd.DataFrame(runs).drop(columns=["class_counts"]), hide_index=True)
        st.caption("Only aggregate statistics are stored in SQLite - never the uploaded beats.")
