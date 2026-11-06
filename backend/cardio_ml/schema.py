"""Data-validation (schema) checks run before data reaches a model.

These checks prevent "silent failures" where malformed ECG data produces
a confident but meaningless classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import MAX_BATCH_SIZE, RAW_FEATURE_COLUMNS

# Physiologic sanity bounds, in samples (at 128-360 Hz) for RR intervals.
RR_MIN_SAMPLES = 1
RR_MAX_SAMPLES = 5000


class SchemaValidationError(ValueError):
    """Raised when an input batch does not match the expected schema."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_beats_dataframe(df: pd.DataFrame, max_rows: int = MAX_BATCH_SIZE) -> pd.DataFrame:
    """Validate a batch of beats and return a numeric copy of the features.

    Checks (all errors are collected and reported together):
      * batch is not empty and not larger than `max_rows`
      * every required raw feature column is present
      * every feature value is numeric and finite (no NaN / inf / text)
      * RR intervals are positive and within a physiologic range
    """
    errors: list[str] = []

    if df is None or len(df) == 0:
        raise SchemaValidationError(["Batch is empty: at least one beat is required."])
    if len(df) > max_rows:
        raise SchemaValidationError([f"Batch has {len(df)} rows; the maximum is {max_rows}."])

    missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns ({len(missing)}): {missing}")
        raise SchemaValidationError(errors)

    feats = df[RAW_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    non_numeric = feats.isna() & df[RAW_FEATURE_COLUMNS].notna()
    if non_numeric.any().any():
        bad_cols = non_numeric.any()[lambda s: s].index.tolist()
        errors.append(f"Non-numeric values found in columns: {bad_cols}")

    nan_counts = df[RAW_FEATURE_COLUMNS].isna().sum()
    if nan_counts.any():
        errors.append(f"Missing (NaN) values: {nan_counts[nan_counts > 0].to_dict()}")

    if np.isinf(feats.to_numpy(dtype=float, na_value=0.0)).any():
        errors.append("Infinite values found in feature columns.")

    for col in ("0_pre-RR", "0_post-RR"):
        vals = feats[col].dropna()
        out_of_range = ((vals < RR_MIN_SAMPLES) | (vals > RR_MAX_SAMPLES)).sum()
        if out_of_range:
            errors.append(
                f"{col}: {int(out_of_range)} value(s) outside the physiologic range "
                f"[{RR_MIN_SAMPLES}, {RR_MAX_SAMPLES}] samples."
            )

    if errors:
        raise SchemaValidationError(errors)
    return feats.astype(float)
