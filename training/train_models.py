"""Train, evaluate and serialize the arrhythmia classifiers.

Run from the repository root:

    python training/train_models.py

Inputs : data/raw/mitbih_features.csv   (MIT-BIH, 360 Hz)
         data/raw/incart_features.csv   (INCART, 257 Hz)  [optional]
Outputs: backend/models/model_<name>.pkl
         backend/models/metrics.json
         frontend/sample_data/*.csv       (demo upload files + EDA sample)
         (then run training/make_report_figures.py for reports/figures/*.png)

Evaluation design
-----------------
Beats from the same patient look alike, so a random train/test split
leaks patient identity and inflates scores. We therefore use the standard
*inter-patient* split of de Chazal et al. (2004): the models are trained
on 22 recordings (DS1) and tested on 22 *different* patients (DS2). The
random split is also reported to show how large the leakage effect is.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from cardio_ml.constants import AAMI_CLASSES, RAW_FEATURE_COLUMNS, REDUNDANT_COLUMNS  # noqa: E402
from cardio_ml.features import (  # noqa: E402
    ENGINEERED_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS, RAW_MODEL_COLUMNS, engineer_features,
)
from cardio_ml.rules import RuleBasedClassifier  # noqa: E402
from cardio_ml.wrappers import StringLabelClassifier  # noqa: E402

RANDOM_STATE = 42
TRAIN_CLASSES = ["N", "SVEB", "VEB", "F"]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "backend" / "models"
SAMPLE_DIR = ROOT / "frontend" / "sample_data"
FIG_DIR = ROOT / "reports" / "figures"

# de Chazal et al. (2004) inter-patient split (paced records 102, 104,
# 107, 217 are already excluded from the dataset, per AAMI).
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def make_models() -> dict[str, tuple[str, object]]:
    """name -> (display name, unfitted estimator)."""
    models = {
        "logistic": ("Logistic Regression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
        ])),
        "rf": ("Random Forest", Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=150, max_depth=18, min_samples_leaf=3,
                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)),
        ])),
        "hgb": ("Gradient Boosting (HistGB)", Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                class_weight="balanced", random_state=RANDOM_STATE)),
        ])),
    }
    try:  # XGBoost is optional: used when installed (as in the reference repo)
        from xgboost import XGBClassifier
        models["xgb"] = ("XGBoost", Pipeline([
            ("clf", StringLabelClassifier(XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                colsample_bytree=0.8, eval_metric="mlogloss", random_state=RANDOM_STATE))),
        ]))
    except ImportError:
        pass  # xgboost not installed - XGBoost model skipped
    return models


def evaluate(y_true, y_pred, proba: np.ndarray | None = None) -> dict:
    labels_present = [c for c in TRAIN_CLASSES if c in set(y_true) | set(y_pred)]
    report = classification_report(y_true, y_pred, labels=AAMI_CLASSES,
                                    output_dict=True, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels_present, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in AAMI_CLASSES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=AAMI_CLASSES).tolist(),
        "labels": AAMI_CLASSES,
    }
    if proba is not None:
        y_abn = (np.asarray(y_true) != "N").astype(int)
        risk = 1 - proba[:, AAMI_CLASSES.index("N")]
        out["abnormal_auc"] = roc_auc_score(y_abn, risk)
    return _round(out)


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def proba_aligned(est, X) -> np.ndarray:
    p = est.predict_proba(X)
    out = np.zeros((len(X), len(AAMI_CLASSES)))
    for j, c in enumerate(est.classes_):
        out[:, AAMI_CLASSES.index(str(c))] = p[:, j]
    return out


def fit_predict(est, X_tr, y_tr, X_te):
    """Fit a fresh copy and return (fitted estimator, predicted labels, aligned proba)."""
    est = clone(est).fit(X_tr, y_tr)
    proba = proba_aligned(est, X_te)
    pred = np.array(AAMI_CLASSES)[proba.argmax(axis=1)]
    return est, pred, proba


def feature_importance(est, X_te, y_te, feature_names, name) -> list[dict]:
    """Permutation importance (model-agnostic) on a stratified DS2 subsample."""
    Xs, _, ys, _ = train_test_split(X_te, y_te, train_size=6000, stratify=y_te,
                                    random_state=RANDOM_STATE)
    scorer = "balanced_accuracy"
    r = permutation_importance(est, Xs, ys, n_repeats=3, random_state=RANDOM_STATE,
                               scoring=scorer, n_jobs=-1)
    order = np.argsort(r.importances_mean)[::-1][:20]
    return [{"feature": feature_names[i], "importance": round(float(r.importances_mean[i]), 5),
             "std": round(float(r.importances_std[i]), 5),
             "engineered": feature_names[i] in ENGINEERED_FEATURE_COLUMNS} for i in order]


def main() -> None:
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. Load & clean ----------------
    df = pd.read_csv(DATA_DIR / "mitbih_features.csv")
    n_raw = len(df)
    df = df.dropna(subset=RAW_FEATURE_COLUMNS + ["type"])
    df = df[(df["0_pre-RR"] > 0) & (df["0_post-RR"] > 0)].reset_index(drop=True)
    n_q = int((df["type"] == "Q").sum())
    # Q (unclassifiable) has only 15 beats in the whole database (8 in DS1),
    # far too few to learn from; with class balancing it produced hundreds
    # of false "Q" predictions. As in most inter-patient studies it is
    # excluded from training and evaluation (the API still reports p_Q = 0).
    df = df[df["type"].isin(TRAIN_CLASSES)].reset_index(drop=True)
    print(f"Loaded {n_raw} beats, removed {n_q} Q beats -> {len(df)} beats, "
          f"{df.record.nunique()} records")

    # ---------------- 2. Feature engineering ----------------
    feats = engineer_features(df, sampling_rate_hz=360)
    y = df["type"].astype(str).to_numpy()
    is_ds1 = df["record"].isin(DS1).to_numpy()
    is_ds2 = df["record"].isin(DS2).to_numpy()
    X_tr, X_te, y_tr, y_te = feats[is_ds1], feats[is_ds2], y[is_ds1], y[is_ds2]
    print(f"DS1 (train): {len(X_tr)} beats | DS2 (test): {len(X_te)} beats")

    results: dict = {"models": {}, "ablation": {}, "random_split": {}, "external_incart": {}}

    # ---------------- 3. Rule-based baseline ----------------
    rules = RuleBasedClassifier()
    r_pred = rules.predict(X_te)
    results["models"]["rules"] = {"display_name": "Rule-based triage (fallback)",
                                  "inter_patient": evaluate(y_te, r_pred, rules.predict_proba(X_te))}

    # ---------------- 4. Train ML models (inter-patient) ----------------
    fitted = {}
    for name, (display, est) in make_models().items():
        t = time.time()
        model, pred, proba = fit_predict(est, X_tr, y_tr, X_te)
        m = evaluate(y_te, pred, proba)
        m["train_seconds"] = round(time.time() - t, 1)
        results["models"][name] = {"display_name": display, "inter_patient": m}
        fitted[name] = model
        print(f"{display:28s} acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f} "
              f"macroF1={m['macro_f1']:.3f} AUC={m['abnormal_auc']:.3f} ({m['train_seconds']}s)")

    # ---------------- 5. Feature-engineering ablation ----------------
    for name in ("logistic", "hgb"):
        est = make_models()[name][1]
        _, pred_raw, proba_raw = fit_predict(est, X_tr[RAW_MODEL_COLUMNS], y_tr, X_te[RAW_MODEL_COLUMNS])
        raw_m = evaluate(y_te, pred_raw, proba_raw)
        eng_m = results["models"][name]["inter_patient"]
        results["ablation"][name] = {
            "raw_only": {k: raw_m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "abnormal_auc")},
            "raw_plus_engineered": {k: eng_m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "abnormal_auc")},
            "raw_only_recall": {c: raw_m["per_class"][c]["recall"] for c in AAMI_CLASSES},
            "engineered_recall": {c: eng_m["per_class"][c]["recall"] for c in AAMI_CLASSES},
        }
        print(f"Ablation {name}: raw bal_acc={raw_m['balanced_accuracy']:.3f} -> "
              f"engineered {eng_m['balanced_accuracy']:.3f}")

    # ---------------- 6. Random (intra-patient) split, to show leakage ----------------
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(feats, y, test_size=0.2, stratify=y,
                                                  random_state=RANDOM_STATE)
    for name in ("logistic", "hgb"):
        _, pred_r, proba_r = fit_predict(make_models()[name][1], Xr_tr, yr_tr, Xr_te)
        m = evaluate(yr_te, pred_r, proba_r)
        results["random_split"][name] = {k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "abnormal_auc")}

    # ---------------- 7. Pick best model, importances ----------------
    ml_names = [n for n in fitted]
    best = max(ml_names, key=lambda n: results["models"][n]["inter_patient"]["balanced_accuracy"])
    results["best_model"] = best
    for name in ml_names:
        results["models"][name]["feature_importance"] = feature_importance(
            fitted[name], X_te, y_te, MODEL_FEATURE_COLUMNS, name)

    # ---------------- 8. External validation on INCART (different hospital, 257 Hz) ----------------
    incart_path = DATA_DIR / "incart_features.csv"
    if incart_path.exists():
        inc = pd.read_csv(incart_path).dropna(subset=RAW_FEATURE_COLUMNS + ["type"])
        inc = inc[(inc["0_pre-RR"] > 0) & (inc["0_post-RR"] > 0)].reset_index(drop=True)
        X_inc = engineer_features(inc, sampling_rate_hz=257)
        y_inc = inc["type"].astype(str).to_numpy()
        for name in ml_names:
            proba = proba_aligned(fitted[name], X_inc)
            pred = np.array(AAMI_CLASSES)[proba.argmax(axis=1)]
            m = evaluate(y_inc, pred, proba)
            results["external_incart"][name] = {k: m[k] for k in
                ("accuracy", "balanced_accuracy", "macro_f1", "abnormal_auc")}
            results["external_incart"][name]["recall"] = {c: m["per_class"][c]["recall"] for c in AAMI_CLASSES}
        # Same best model but WITHOUT the correct sampling-rate conversion:
        X_wrong = engineer_features(inc, sampling_rate_hz=360)
        proba = proba_aligned(fitted[best], X_wrong)
        pred = np.array(AAMI_CLASSES)[proba.argmax(axis=1)]
        m = evaluate(y_inc, pred, proba)
        results["external_incart"][f"{best}_wrong_sampling_rate"] = {k: m[k] for k in
            ("accuracy", "balanced_accuracy", "macro_f1", "abnormal_auc")}
        print(f"INCART external ({best}): bal_acc={results['external_incart'][best]['balanced_accuracy']:.3f}")

    # ---------------- 9. Serialize ----------------
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, model in fitted.items():
        bundle = {
            "name": name,
            "display_name": results["models"][name]["display_name"],
            "estimator": model,
            "feature_names": MODEL_FEATURE_COLUMNS,
            "classes": AAMI_CLASSES,
            "metrics": results["models"][name],
            "trained_at": trained_at,
            "training_data": "MIT-BIH Arrhythmia DB (DS1, 22 records)",
        }
        joblib.dump(bundle, MODEL_DIR / f"model_{name}.pkl", compress=3)
    results["dataset"] = {
        "name": "MIT-BIH Arrhythmia Database (Sakib et al. features)",
        "n_beats": int(len(df)), "n_records": int(df.record.nunique()),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "class_counts": df["type"].value_counts().reindex(AAMI_CLASSES, fill_value=0).astype(int).to_dict(),
        "train_class_counts": pd.Series(y_tr).value_counts().reindex(AAMI_CLASSES, fill_value=0).astype(int).to_dict(),
        "test_class_counts": pd.Series(y_te).value_counts().reindex(AAMI_CLASSES, fill_value=0).astype(int).to_dict(),
        "n_q_beats_excluded": n_q,
        "n_raw_features": len(RAW_FEATURE_COLUMNS),
        "redundant_columns_dropped": REDUNDANT_COLUMNS,
        "n_engineered_features": len(ENGINEERED_FEATURE_COLUMNS),
        "n_model_features": len(MODEL_FEATURE_COLUMNS),
        "train_records": DS1, "test_records": DS2,
    }
    results["trained_at"] = trained_at
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=int))

    # ---------------- 10. Sample data for the dashboard & tests ----------------
    write_samples(df, feats)
    print(f"Done in {time.time() - t0:.0f}s. Best model: {best}")


def write_samples(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    cols = ["record", "type"] + RAW_FEATURE_COLUMNS
    # Demo upload files: whole unseen (DS2) recordings, in time order.
    for rec, desc in [(100, "mostly_normal"), (208, "frequent_PVC_fusion"),
                      (232, "frequent_SVEB"), (213, "mixed_ectopy")]:
        df.loc[df.record == rec, cols].to_csv(SAMPLE_DIR / f"record_{rec}_{desc}.csv", index=False)
    # Multi-patient batch (> 10,000 beats) for the performance requirement.
    # All six are DS2 (test) patients, so the dashboard demo is never on training data.
    df.loc[df.record.isin([200, 210, 213, 219, 221, 233]), cols].to_csv(
        SAMPLE_DIR / "multi_patient_batch.csv", index=False)
    # EDA sample from the TRAINING records only (stratified, <= 2,500 beats per class),
    # with engineered features, for the dashboard's exploratory charts.
    eda = pd.concat([df[["record", "type"]], feats], axis=1)
    eda = eda[eda["record"].isin(DS1)]
    idx = [i for _, g in eda.groupby("type")
           for i in g.sample(min(len(g), 2500), random_state=RANDOM_STATE).index]
    eda.loc[sorted(idx)].to_csv(SAMPLE_DIR / "eda_sample.csv", index=False)


if __name__ == "__main__":
    main()
