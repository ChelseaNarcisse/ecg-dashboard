"""Estimator wrappers that must be importable when a .pkl is loaded."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


class StringLabelClassifier(ClassifierMixin, BaseEstimator):
    """Lets estimators that need integer labels (e.g. XGBoost) accept the
    string AAMI labels ("N", "SVEB", ...) like scikit-learn models do."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.encoder_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))
