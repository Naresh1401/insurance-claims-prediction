"""Risk-based pricing model: Pure Premium = Frequency × Severity (GLM framework)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor


class FrequencyModel(BaseEstimator, RegressorMixin):
    """Poisson GLM for claim frequency prediction."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = None

    def fit(self, X, y):
        self.model = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)


class SeverityGLM(BaseEstimator, RegressorMixin):
    """Gamma GLM for claim severity (conditional on a claim occurring)."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = None

    def fit(self, X, y):
        # Gamma requires strictly positive targets
        mask = y > 0
        self.model = GammaRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self.model.fit(X[mask], y[mask])
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)


class RiskPricer(BaseEstimator, RegressorMixin):
    """
    Two-part actuarial pricing model:
      Pure Premium = E[Frequency] × E[Severity]

    Frequency modeled with Poisson GLM.
    Severity modeled with Gamma GLM.
    """

    def __init__(
        self,
        freq_alpha: float = 1.0,
        sev_alpha: float = 1.0,
        expense_loading: float = 0.15,
        profit_margin: float = 0.05,
        max_iter: int = 1000,
    ):
        self.freq_alpha = freq_alpha
        self.sev_alpha = sev_alpha
        self.expense_loading = expense_loading
        self.profit_margin = profit_margin
        self.max_iter = max_iter
        self.freq_model = None
        self.sev_model = None
        self.tweedie_model = None

    def fit(self, X, y_freq, y_sev):
        """
        Fit both frequency and severity models.

        Parameters
        ----------
        X : array-like - Feature matrix
        y_freq : array-like - Claim counts (frequency)
        y_sev : array-like - Claim amounts (severity, positive values only)
        """
        self.freq_model = FrequencyModel(alpha=self.freq_alpha, max_iter=self.max_iter)
        self.freq_model.fit(X, y_freq)

        self.sev_model = SeverityGLM(alpha=self.sev_alpha, max_iter=self.max_iter)
        self.sev_model.fit(X, y_sev)

        # Also fit a direct Tweedie model for comparison
        self.tweedie_model = TweedieRegressor(
            power=1.5, alpha=1.0, max_iter=self.max_iter, link="log"
        )
        pos_mask = y_sev > 0
        self.tweedie_model.fit(X[pos_mask], y_sev[pos_mask])

        return self

    def predict_pure_premium(self, X) -> np.ndarray:
        """Pure premium = frequency × severity."""
        freq = self.freq_model.predict(X)
        sev = self.sev_model.predict(X)
        return freq * sev

    def predict_gross_premium(self, X) -> np.ndarray:
        """Gross premium with expense and profit loadings."""
        pure = self.predict_pure_premium(X)
        return pure / (1 - self.expense_loading - self.profit_margin)

    def predict(self, X):
        """Default predict returns gross premium."""
        return self.predict_gross_premium(X)

    def premium_adequacy(self, X, actual_premiums: np.ndarray) -> dict:
        """Analyze premium adequacy: compare predicted vs actual premiums."""
        predicted = self.predict_gross_premium(X)
        pure = self.predict_pure_premium(X)

        return {
            "predicted_premium_mean": float(predicted.mean()),
            "actual_premium_mean": float(actual_premiums.mean()),
            "adequacy_ratio": float(actual_premiums.mean() / predicted.mean()),
            "pure_premium_mean": float(pure.mean()),
            "loading_factor": float(predicted.mean() / pure.mean()),
            "underprice_pct": float((actual_premiums < pure).mean()),
            "overprice_pct": float((actual_premiums > predicted).mean()),
        }

    def loss_ratio_by_segment(
        self, X, actual_claims: np.ndarray, actual_premiums: np.ndarray, segments: np.ndarray
    ) -> pd.DataFrame:
        """Calculate loss ratio by segment."""
        df = pd.DataFrame(
            {
                "segment": segments,
                "actual_claims": actual_claims,
                "actual_premium": actual_premiums,
                "predicted_premium": self.predict_gross_premium(X),
            }
        )
        agg = df.groupby("segment").agg(
            total_claims=("actual_claims", "sum"),
            total_actual_premium=("actual_premium", "sum"),
            total_predicted_premium=("predicted_premium", "sum"),
            count=("segment", "size"),
        )
        agg["actual_loss_ratio"] = agg["total_claims"] / agg["total_actual_premium"]
        agg["predicted_loss_ratio"] = agg["total_claims"] / agg["total_predicted_premium"]
        return agg.round(4)
