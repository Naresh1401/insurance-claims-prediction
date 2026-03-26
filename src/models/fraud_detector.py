"""Fraud detection models combining supervised and unsupervised approaches."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class IsolationForestDetector(BaseEstimator, ClassifierMixin):
    """Unsupervised anomaly detection for fraud."""

    def __init__(
        self,
        n_estimators: int = 300,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def fit(self, X, y=None):
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        return self

    def predict(self, X):
        raw = self.model.predict(X)
        return (raw == -1).astype(int)  # -1 = anomaly → 1 = fraud

    def anomaly_score(self, X) -> np.ndarray:
        return -self.model.score_samples(X)

    def predict_proba(self, X):
        scores = self.anomaly_score(X)
        # Normalize to [0, 1] range
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return np.column_stack([1 - scores_norm, scores_norm])


class XGBFraudDetector(BaseEstimator, ClassifierMixin):
    """XGBoost with SMOTE for imbalanced fraud detection."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: float = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.pipeline = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        if self.scale_pos_weight is None:
            neg, pos = np.bincount(y.astype(int))
            spw = neg / max(pos, 1)
        else:
            spw = self.scale_pos_weight

        xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.pipeline = ImbPipeline(
            [
                ("smote", SMOTE(random_state=self.random_state, sampling_strategy=0.3)),
                ("xgb", xgb),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def anomaly_score(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]


class LGBMFraudDetector(BaseEstimator, ClassifierMixin):
    """LightGBM fraud detector with focal loss for class imbalance."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.pipeline = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        neg, pos = np.bincount(y.astype(int))
        lgbm = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            is_unbalance=True,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.pipeline = ImbPipeline(
            [
                ("smote", SMOTE(random_state=self.random_state, sampling_strategy=0.3)),
                ("lgbm", lgbm),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def anomaly_score(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]


FRAUD_MODELS = {
    "isolation_forest": IsolationForestDetector,
    "xgboost_smote": XGBFraudDetector,
    "lightgbm": LGBMFraudDetector,
}
