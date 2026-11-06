"""Feature engineering for ECG beat classification.

The raw Kaggle features are measured in *samples* and are absolute
amplitudes. Cardiologists, however, reason about beats *relative to the
patient's own rhythm*: a premature beat arrives early compared with the
beats around it, and a ventricular beat has a wide QRS complex measured
in milliseconds. The functions below turn the raw columns into those
clinically meaningful features.

Engineered feature groups
-------------------------
1. Unit conversion      samples -> milliseconds using the sampling rate
2. Rhythm (RR) features  RR ratio, compensatory pause, heart rate,
                         RR normalised to the patient's local rhythm
3. Interval features     QRS / QT / PQ / ST durations, Bazett QTc, wide-QRS flag
4. Morphology features   QRS amplitude, R/S and T/R ratios, QRS energy & slope,
                         cross-lead R-peak difference
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_SAMPLING_RATE_HZ,
    LEADS,
    RAW_FEATURE_COLUMNS,
    REDUNDANT_COLUMNS,
)

EPS = 1e-6
LOCAL_RR_WINDOW = 10  # beats used for the patient's "local" rhythm baseline

INTERVALS = ["qrs_interval", "pq_interval", "qt_interval", "st_interval"]

ENGINEERED_FEATURE_COLUMNS = [
    # rhythm
    "pre_rr_ms", "post_rr_ms", "rr_ratio", "rr_diff_ms", "heart_rate_bpm",
    "local_rr_mean_ms", "rr_norm_local", "rr_norm_record", "post_rr_norm_local",
    # intervals (both leads)
    *[f"{lead}_{iv.replace('_interval', '')}_ms" for lead in LEADS for iv in INTERVALS],
    "qtc_bazett_ms", "wide_qrs_flag",
    # morphology (both leads)
    *[f"{lead}_{name}" for lead in LEADS for name in
      ("qrs_amplitude", "r_s_ratio", "t_r_ratio", "qrs_energy", "qrs_slope")],
    "lead_r_diff",
]

# Raw columns kept for modeling (redundant duplicates removed).
RAW_MODEL_COLUMNS = [c for c in RAW_FEATURE_COLUMNS if c not in REDUNDANT_COLUMNS]

# Final, ordered list of columns fed to the models.
MODEL_FEATURE_COLUMNS = RAW_MODEL_COLUMNS + ENGINEERED_FEATURE_COLUMNS


def samples_to_ms(samples: pd.Series | np.ndarray, sampling_rate_hz: float) -> pd.Series:
    """Convert a duration measured in samples into milliseconds."""
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    return samples * (1000.0 / sampling_rate_hz)


def bazett_qtc(qt_ms: pd.Series, rr_ms: pd.Series) -> pd.Series:
    """Heart-rate-corrected QT interval: QTc = QT / sqrt(RR in seconds)."""
    rr_s = np.clip(rr_ms / 1000.0, 0.2, 3.0)  # clip to physiologic range
    return qt_ms / np.sqrt(rr_s)


def safe_ratio(num: pd.Series, den: pd.Series, clip: float = 10.0) -> pd.Series:
    """Ratio that never divides by zero and is clipped to +/- `clip`."""
    den = den.where(den.abs() > EPS, np.sign(den).replace(0, 1) * EPS)
    return (num / den).clip(-clip, clip)


def local_rr_baseline(pre_rr_ms: pd.Series, groups: pd.Series | None,
                      window: int = LOCAL_RR_WINDOW) -> pd.Series:
    """Causal rolling mean of the *previous* `window` RR intervals.

    Only past beats are used (shift(1)), which mimics a wearable device
    that must decide about the current beat in real time. The first beat
    of each recording falls back to its own RR interval.
    """
    def _roll(s: pd.Series) -> pd.Series:
        baseline = s.shift(1).rolling(window, min_periods=1).mean()
        return baseline.fillna(s)

    if groups is None:
        return _roll(pre_rr_ms)
    return pre_rr_ms.groupby(groups, sort=False).transform(_roll)


def engineer_features(df: pd.DataFrame,
                      sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
                      group_col: str | None = "record") -> pd.DataFrame:
    """Return a DataFrame with the columns in MODEL_FEATURE_COLUMNS.

    Parameters
    ----------
    df : DataFrame containing at least RAW_FEATURE_COLUMNS, with beats in
         time order within each recording.
    sampling_rate_hz : sampling frequency the interval columns were measured at.
    group_col : column identifying the recording/patient. Rhythm baselines
         are computed within each group. If the column is missing the whole
         batch is treated as one recording.
    """
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required raw feature columns: {missing}")

    raw = df[RAW_FEATURE_COLUMNS].astype(float).reset_index(drop=True)
    groups = None
    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str).reset_index(drop=True)

    out = pd.DataFrame(index=raw.index)

    # --- 1. Rhythm features -------------------------------------------------
    pre = samples_to_ms(raw["0_pre-RR"], sampling_rate_hz)
    post = samples_to_ms(raw["0_post-RR"], sampling_rate_hz)
    out["pre_rr_ms"] = pre
    out["post_rr_ms"] = post
    out["rr_ratio"] = safe_ratio(pre, post)            # <1 => premature beat
    out["rr_diff_ms"] = post - pre                     # >0 => compensatory pause
    out["heart_rate_bpm"] = (60000.0 / pre.clip(lower=150)).clip(upper=400)

    local = local_rr_baseline(pre, groups)
    out["local_rr_mean_ms"] = local
    out["rr_norm_local"] = safe_ratio(pre, local)
    out["post_rr_norm_local"] = safe_ratio(post, local)
    if groups is None:
        record_median = pd.Series(pre.median(), index=pre.index)
    else:
        record_median = pre.groupby(groups, sort=False).transform("median")
    out["rr_norm_record"] = safe_ratio(pre, record_median)

    # --- 2. Interval features ----------------------------------------------
    for lead in LEADS:
        for iv in INTERVALS:
            name = f"{lead}_{iv.replace('_interval', '')}_ms"
            out[name] = samples_to_ms(raw[f"{lead}_{iv}"], sampling_rate_hz)
    out["qtc_bazett_ms"] = bazett_qtc(out["0_qt_ms"], pre).clip(upper=2000)
    out["wide_qrs_flag"] = (out["0_qrs_ms"] > 120).astype(float)

    # --- 3. Morphology features --------------------------------------------
    for lead in LEADS:
        r, s, t = raw[f"{lead}_rPeak"], raw[f"{lead}_sPeak"], raw[f"{lead}_tPeak"]
        morph = raw[[f"{lead}_qrs_morph{i}" for i in range(5)]]
        out[f"{lead}_qrs_amplitude"] = r - s
        out[f"{lead}_r_s_ratio"] = safe_ratio(r.abs(), s.abs())
        out[f"{lead}_t_r_ratio"] = safe_ratio(t, r)
        out[f"{lead}_qrs_energy"] = (morph ** 2).sum(axis=1)
        out[f"{lead}_qrs_slope"] = raw[f"{lead}_qrs_morph4"] - raw[f"{lead}_qrs_morph0"]
    out["lead_r_diff"] = raw["0_rPeak"] - raw["1_rPeak"]

    result = pd.concat([raw[RAW_MODEL_COLUMNS], out[ENGINEERED_FEATURE_COLUMNS]], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result[MODEL_FEATURE_COLUMNS]
