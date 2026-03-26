"""Claim severity prediction models (regression)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import TweedieRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class GammaSeverityModel(BaseEstimator, RegressorMixin):
    """Tweedie/Gamma GLM for claim severity with zero-inflation handling."""

    def __init__(self, power: float = 1.5, alpha: float = 1.0, max_iter: int = 1000):
        self.power = power
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = None
        self.zero_threshold_ = None

    def fit(self, X, y):
        # Separate zero and non-zero claims
        nonzero_mask = y > 0
        self.zero_threshold_ = 1 - nonzero_mask.mean()

        self.model = TweedieRegressor(
            power=self.power, alpha=self.alpha, max_iter=self.max_iter, link="log"
        )
        # Fit only on positive claims
        if nonzero_mask.sum() > 0:
            self.model.fit(X[nonzero_mask], y[nonzero_mask])
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)


class RandomForestSeverity(BaseEstimator, RegressorMixin):
    """Random Forest regressor for severity with zero handling."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)


class XGBSeverity(BaseEstimator, RegressorMixin):
    """XGBoost with Tweedie objective for zero-inflated severity."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        tweedie_variance_power: float = 1.5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.tweedie_variance_power = tweedie_variance_power
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = XGBRegressor(
            objective=f"reg:tweedie",
            tweedie_variance_power=self.tweedie_variance_power,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)


class LGBMSeverity(BaseEstimator, RegressorMixin):
    """LightGBM with Tweedie objective."""

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
        self.model = None

    def fit(self, X, y):
        self.model = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)


SEVERITY_MODELS = {
    "gamma_glm": GammaSeverityModel,
    "random_forest": RandomForestSeverity,
    "xgboost": XGBSeverity,
    "lightgbm": LGBMSeverity,
}
