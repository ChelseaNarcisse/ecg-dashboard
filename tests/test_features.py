"""Unit tests for feature engineering (cardio_ml.features)."""

import numpy as np
import pandas as pd
import pytest

from cardio_ml.features import (
    ENGINEERED_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS, bazett_qtc, engineer_features,
    local_rr_baseline, safe_ratio, samples_to_ms,
)
from cardio_ml.constants import REDUNDANT_COLUMNS


def test_samples_to_ms_at_360hz():
    assert samples_to_ms(pd.Series([360.0]), 360).iloc[0] == pytest.approx(1000.0)


def test_samples_to_ms_rejects_bad_rate():
    with pytest.raises(ValueError):
        samples_to_ms(pd.Series([1.0]), 0)


def test_bazett_qtc_equals_qt_at_one_second_rr():
    qtc = bazett_qtc(pd.Series([400.0]), pd.Series([1000.0]))
    assert qtc.iloc[0] == pytest.approx(400.0)


def test_safe_ratio_never_divides_by_zero():
    r = safe_ratio(pd.Series([1.0, 1.0]), pd.Series([0.0, 2.0]))
    assert np.isfinite(r).all()
    assert r.iloc[1] == pytest.approx(0.5)


def test_local_rr_baseline_is_causal():
    """The baseline for a beat must only use *previous* beats."""
    rr = pd.Series([800.0, 800.0, 800.0, 400.0])
    base = local_rr_baseline(rr, groups=None)
    assert base.iloc[3] == pytest.approx(800.0)  # the early beat is not in its own baseline


def test_local_rr_baseline_respects_groups():
    rr = pd.Series([800.0, 800.0, 500.0, 500.0])
    groups = pd.Series(["a", "a", "b", "b"])
    base = local_rr_baseline(rr, groups)
    assert base.iloc[2] == pytest.approx(500.0)  # patient b does not inherit patient a's rhythm


def test_output_has_expected_columns_and_no_missing(sample_df):
    feats = engineer_features(sample_df)
    assert list(feats.columns) == MODEL_FEATURE_COLUMNS
    assert len(feats) == len(sample_df)
    assert not feats.isna().any().any()
    assert np.isfinite(feats.to_numpy()).all()


def test_redundant_columns_are_dropped(sample_df):
    feats = engineer_features(sample_df)
    assert not set(REDUNDANT_COLUMNS) & set(feats.columns)
    assert set(ENGINEERED_FEATURE_COLUMNS) <= set(feats.columns)


def test_premature_beat_has_rr_ratio_below_one(sample_df):
    df = sample_df.copy()
    df.loc[5, "0_pre-RR"] = 150     # beat arrives early ...
    df.loc[5, "0_post-RR"] = 450    # ... followed by a compensatory pause
    feats = engineer_features(df)
    assert feats.loc[5, "rr_ratio"] < 1
    assert feats.loc[5, "rr_diff_ms"] > 0
    assert feats.loc[5, "rr_norm_local"] < 0.9


def test_sampling_rate_changes_millisecond_features(sample_df):
    at_360 = engineer_features(sample_df, sampling_rate_hz=360)
    at_180 = engineer_features(sample_df, sampling_rate_hz=180)
    assert np.allclose(at_180["0_qrs_ms"], 2 * at_360["0_qrs_ms"])
    # ratios are unit-free, so they must NOT depend on the sampling rate
    assert np.allclose(at_180["rr_ratio"], at_360["rr_ratio"])


def test_wide_qrs_flag(sample_df):
    df = sample_df.copy()
    df.loc[0, "0_qrs_interval"] = 60   # 60 samples @360 Hz = 167 ms -> wide
    df.loc[1, "0_qrs_interval"] = 20   # 56 ms -> narrow
    feats = engineer_features(df)
    assert feats.loc[0, "wide_qrs_flag"] == 1
    assert feats.loc[1, "wide_qrs_flag"] == 0


def test_works_without_record_column(sample_df):
    feats = engineer_features(sample_df.drop(columns=["record"]))
    assert len(feats) == len(sample_df)


def test_missing_raw_column_raises(sample_df):
    with pytest.raises(KeyError):
        engineer_features(sample_df.drop(columns=["0_rPeak"]))
