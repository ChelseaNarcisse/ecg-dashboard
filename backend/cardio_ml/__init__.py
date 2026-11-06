"""cardio_ml: feature engineering, schema validation, and model utilities
for ECG beat classification (AAMI classes N, SVEB, VEB, F, Q).

This package is shared by the training script (training/train_models.py)
and the FastAPI backend (backend/app.py), so the exact same feature
engineering runs at training time and at inference time.
"""

from .constants import AAMI_CLASSES, CLASS_DESCRIPTIONS, RAW_FEATURE_COLUMNS
from .features import engineer_features, ENGINEERED_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from .schema import SchemaValidationError, validate_beats_dataframe

__all__ = [
    "AAMI_CLASSES",
    "CLASS_DESCRIPTIONS",
    "RAW_FEATURE_COLUMNS",
    "ENGINEERED_FEATURE_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "engineer_features",
    "SchemaValidationError",
    "validate_beats_dataframe",
]
