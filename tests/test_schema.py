"""Data-validation (schema) tests: malformed ECG data must never reach a model."""

import numpy as np
import pandas as pd
import pytest

from cardio_ml.schema import SchemaValidationError, validate_beats_dataframe


def test_valid_batch_passes(sample_df):
    out = validate_beats_dataframe(sample_df)
    assert len(out) == len(sample_df)


def test_empty_batch_rejected():
    with pytest.raises(SchemaValidationError, match="empty"):
        validate_beats_dataframe(pd.DataFrame())


def test_too_many_rows_rejected(sample_df):
    with pytest.raises(SchemaValidationError, match="maximum"):
        validate_beats_dataframe(sample_df, max_rows=10)


def test_missing_column_rejected(sample_df):
    with pytest.raises(SchemaValidationError, match="Missing required columns"):
        validate_beats_dataframe(sample_df.drop(columns=["0_qt_interval", "1_tPeak"]))


def test_nan_values_rejected(sample_df):
    df = sample_df.copy()
    df.loc[3, "0_rPeak"] = np.nan
    with pytest.raises(SchemaValidationError, match="NaN"):
        validate_beats_dataframe(df)


def test_text_values_rejected(sample_df):
    df = sample_df.astype({"0_pPeak": object})
    df.loc[0, "0_pPeak"] = "abc"
    with pytest.raises(SchemaValidationError, match="Non-numeric"):
        validate_beats_dataframe(df)


def test_infinite_values_rejected(sample_df):
    df = sample_df.copy()
    df.loc[0, "0_tPeak"] = np.inf
    with pytest.raises(SchemaValidationError, match="Infinite"):
        validate_beats_dataframe(df)


@pytest.mark.parametrize("bad_rr", [0, -5, 99999])
def test_rr_outside_physiologic_range_rejected(sample_df, bad_rr):
    df = sample_df.copy()
    df.loc[0, "0_pre-RR"] = bad_rr
    with pytest.raises(SchemaValidationError, match="physiologic"):
        validate_beats_dataframe(df)


def test_all_errors_reported_together(sample_df):
    df = sample_df.copy()
    df.loc[0, "0_rPeak"] = np.nan
    df.loc[1, "0_post-RR"] = -1
    with pytest.raises(SchemaValidationError) as exc:
        validate_beats_dataframe(df)
    assert len(exc.value.errors) == 2
