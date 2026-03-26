"""FastAPI application for insurance claims prediction."""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Insurance Claims Analytics API",
    description="ML-powered claims severity prediction, fraud detection, and risk-based pricing.",
    version="1.0.0",
)

MODEL_DIR = Path("data/models")

# ── Lazy model loading ────────────────────────────────────────────────

_models: dict = {}


def _load_model(name: str):
    if name not in _models:
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"Model '{name}' not found. Run training first.")
        _models[name] = joblib.load(path)
    return _models[name]


# ── Request / Response schemas ────────────────────────────────────────


class ClaimFeatures(BaseModel):
    policyholder_age: int = Field(..., ge=18, le=100)
    coverage_amount: float = Field(..., gt=0)
    premium: float = Field(..., gt=0)
    deductible: float = Field(..., ge=0)
    years_as_customer: int = Field(..., ge=0)
    prior_claims_count: int = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    police_report_filed: int = Field(..., ge=0, le=1)
    witnesses: int = Field(..., ge=0)
    injury_severity_num: int = Field(..., ge=0, le=4)
    property_damage_num: int = Field(..., ge=0, le=4)
    policy_type: Optional[str] = "auto"

    model_config = {"json_schema_extra": {
        "examples": [{
            "policyholder_age": 42,
            "coverage_amount": 50000.0,
            "premium": 1800.0,
            "deductible": 500.0,
            "years_as_customer": 5,
            "prior_claims_count": 1,
            "credit_score": 720,
            "police_report_filed": 1,
            "witnesses": 2,
            "injury_severity_num": 1,
            "property_damage_num": 2,
            "policy_type": "auto",
        }]
    }}


class SeverityResponse(BaseModel):
    predicted_amount: float
    model_used: str


class FraudResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    anomaly_score: float
    risk_level: str


class PremiumResponse(BaseModel):
    pure_premium: float
    gross_premium: float
    adequacy_ratio: Optional[float] = None


# ── Feature vector construction ──────────────────────────────────────


def _build_feature_vector(claim: ClaimFeatures) -> np.ndarray:
    """Build the numeric feature vector expected by models."""
    loss_ratio = claim.premium / claim.coverage_amount if claim.coverage_amount > 0 else 0
    claims_per_year = claim.prior_claims_count / max(claim.years_as_customer, 1)
    is_auto = int(claim.policy_type == "auto")
    is_home = int(claim.policy_type == "home")
    is_health = int(claim.policy_type == "health")
    is_life = int(claim.policy_type == "life")

    vec = [
        claim.policyholder_age,
        claim.coverage_amount,
        claim.premium,
        claim.deductible,
        claim.years_as_customer,
        claim.prior_claims_count,
        claim.credit_score,
        claim.police_report_filed,
        claim.witnesses,
        claim.injury_severity_num,
        claim.property_damage_num,
        min(loss_ratio * 20, 20),                      # loss_ratio_clipped proxy
        0,                                              # claim_to_coverage (unknown at prediction)
        claim.premium / claim.coverage_amount,          # premium_to_coverage
        claim.deductible / claim.coverage_amount,       # deductible_ratio
        claims_per_year,
        int(claims_per_year > 0.5),                     # is_frequent_claimant
        int(claim.prior_claims_count > 3),              # high_prior_claims
        int(claim.years_as_customer < 2),               # new_customer
        int(claim.years_as_customer >= 10),             # loyal_customer
        0.5,                                            # risk_score placeholder
        claim.policyholder_age * is_auto,
        claim.policyholder_age * is_home,
        claim.policyholder_age * is_health,
        claim.policyholder_age * is_life,
        claim.credit_score * claim.prior_claims_count,
        claim.coverage_amount * claim.injury_severity_num,
        claim.witnesses * claim.police_report_filed,
        0, 0, 0, 0,                                    # seasonal placeholders
        0, 0,                                           # high_claim_season, is_weekend
        0, 0, 0, 0, 0, 0, 0,                           # target-encoding placeholders
    ]
    return np.array(vec, dtype=np.float32).reshape(1, -1)


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/health")
def health():
    models_available = [p.stem for p in MODEL_DIR.glob("*.pkl")] if MODEL_DIR.exists() else []
    return {"status": "healthy", "models_loaded": len(_models), "models_available": models_available}


@app.post("/predict/severity", response_model=SeverityResponse)
def predict_severity(claim: ClaimFeatures, model_name: str = "xgboost"):
    model = _load_model(f"severity_{model_name}")
    X = _build_feature_vector(claim)
    amount = float(model.predict(X)[0])
    return SeverityResponse(predicted_amount=round(max(amount, 0), 2), model_used=model_name)


@app.post("/detect/fraud", response_model=FraudResponse)
def detect_fraud(claim: ClaimFeatures, model_name: str = "xgboost_smote"):
    model = _load_model(f"fraud_{model_name}")
    X = _build_feature_vector(claim)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0, 1])
    score = float(model.anomaly_score(X)[0])

    if proba < 0.3:
        risk = "low"
    elif proba < 0.6:
        risk = "medium"
    elif proba < 0.8:
        risk = "high"
    else:
        risk = "critical"

    return FraudResponse(
        is_fraud=bool(pred), fraud_probability=round(proba, 4),
        anomaly_score=round(score, 4), risk_level=risk,
    )


@app.post("/price/premium", response_model=PremiumResponse)
def price_premium(claim: ClaimFeatures):
    pricer = _load_model("risk_pricer")
    X = _build_feature_vector(claim)
    pure = float(pricer.predict_pure_premium(X)[0])
    gross = float(pricer.predict_gross_premium(X)[0])
    adequacy = claim.premium / gross if gross > 0 else None

    return PremiumResponse(
        pure_premium=round(max(pure, 0), 2),
        gross_premium=round(max(gross, 0), 2),
        adequacy_ratio=round(adequacy, 4) if adequacy else None,
    )
