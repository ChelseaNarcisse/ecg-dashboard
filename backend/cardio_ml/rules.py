"""Rule-based triage fallback ("graceful degradation" from the proposal).

If the ML model is unavailable or being retrained, the API can still
return a transparent, threshold-based classification built from the
engineered features. It is also a useful *baseline*: the ML models
should clearly beat it.

Rules (applied in order):
  1. Wide QRS (> 120 ms) AND early beat (RR < 90% of local rhythm) -> VEB
  2. Narrow QRS AND early beat (RR < 85% of local rhythm)          -> SVEB
  3. Otherwise                                                     -> N
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import AAMI_CLASSES

WIDE_QRS_MS = 120.0
VEB_PREMATURITY = 0.90
SVEB_PREMATURITY = 0.85
RULE_CONFIDENCE = 0.75


class RuleBasedClassifier:
    """Mimics the scikit-learn predict / predict_proba interface."""

    classes_ = np.array(AAMI_CLASSES)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        wide = features["0_qrs_ms"] > WIDE_QRS_MS
        early_v = features["rr_norm_local"] < VEB_PREMATURITY
        early_s = features["rr_norm_local"] < SVEB_PREMATURITY
        labels = np.where(wide & early_v, "VEB", np.where(~wide & early_s, "SVEB", "N"))
        return labels

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        labels = self.predict(features)
        n_classes = len(AAMI_CLASSES)
        other = (1.0 - RULE_CONFIDENCE) / (n_classes - 1)
        proba = np.full((len(labels), n_classes), other)
        for i, cls in enumerate(AAMI_CLASSES):
            proba[labels == cls, :] = other
            proba[labels == cls, i] = RULE_CONFIDENCE
        return proba
