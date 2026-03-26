"""Train all models: severity, fraud detection, and risk pricing."""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.data.generate_data import generate_claims
from src.features.feature_engineering import engineer_features
from src.models.severity_model import SEVERITY_MODELS
from src.models.fraud_detector import FRAUD_MODELS
from src.models.risk_pricer import RiskPricer

warnings.filterwarnings("ignore")

MODEL_DIR = Path("data/models")
DATA_DIR = Path("data/processed")

# Numeric feature columns used across models
FEATURE_COLS = [
    "policyholder_age",
    "coverage_amount",
    "premium",
    "deductible",
    "years_as_customer",
    "prior_claims_count",
    "credit_score",
    "police_report_filed",
    "witnesses",
    "injury_severity_num",
    "property_damage_num",
    "loss_ratio_clipped",
    "claim_to_coverage",
    "premium_to_coverage",
    "deductible_ratio",
    "claims_per_year",
    "is_frequent_claimant",
    "high_prior_claims",
    "new_customer",
    "loyal_customer",
    "risk_score",
    "age_x_auto",
    "age_x_home",
    "age_x_health",
    "age_x_life",
    "credit_x_claims",
    "coverage_x_injury",
    "witnesses_x_report",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "high_claim_season",
    "is_weekend",
]

# Add target-encoded columns
TARGET_ENC_COLS = [
    "gender_te",
    "marital_status_te",
    "policy_type_te",
    "region_te",
    "incident_type_te",
    "injury_severity_te",
    "property_damage_extent_te",
]


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Gini coefficient (2*AUC - 1 for binary, or Lorenz-based for continuous)."""
    n = len(y_true)
    arr = np.array([y_true, y_pred]).T
    arr = arr[arr[:, 1].argsort()]
    total = y_true.sum()
    cumulative = np.cumsum(arr[:, 0]) / total
    gini_sum = cumulative.sum() / n
    return 1 - 2 * gini_sum


def time_based_split(df: pd.DataFrame, date_col: str = "claim_date", test_frac: float = 0.2):
    """Split data by time — train on older claims, test on newer."""
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix from engineered DataFrame."""
    cols = [c for c in FEATURE_COLS + TARGET_ENC_COLS if c in df.columns]
    X = df[cols].fillna(0).values.astype(np.float32)
    return X


def train_severity_models(X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate all severity models."""
    results = {}
    models = {}
    for name, ModelClass in SEVERITY_MODELS.items():
        print(f"  Training severity model: {name}")
        model = ModelClass()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}
        models[name] = model
        print(f"    MAE={mae:.2f}, RMSE={rmse:.2f}")
    return models, results


def train_fraud_models(X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate all fraud detection models."""
    results = {}
    models = {}
    for name, ModelClass in FRAUD_MODELS.items():
        print(f"  Training fraud model: {name}")
        model = ModelClass()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, proba)
        results[name] = {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "AUC": round(auc, 4),
        }
        models[name] = model
        print(f"    P={precision:.4f} R={recall:.4f} F1={f1:.4f} AUC={auc:.4f}")
    return models, results


def train_pricing_model(X_train, y_freq_train, y_sev_train, X_test, y_sev_test, premiums_test) -> tuple:
    """Train risk pricing model."""
    print("  Training risk pricing model")
    pricer = RiskPricer()
    pricer.fit(X_train, y_freq_train, y_sev_train)

    predicted = pricer.predict_gross_premium(X_test)
    gini = gini_coefficient(y_sev_test, predicted)
    adequacy = pricer.premium_adequacy(X_test, premiums_test)

    results = {
        "Gini": round(gini, 4),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in adequacy.items()},
    }
    print(f"    Gini={gini:.4f}, Adequacy={adequacy['adequacy_ratio']:.4f}")
    return pricer, results


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Generate & engineer ---
    print("Generating data...")
    df = generate_claims()
    print("Engineering features...")
    df_feat, enc_maps = engineer_features(df)
    df_feat.to_csv(DATA_DIR / "claims_featured.csv", index=False)

    # --- Time-based split ---
    train_df, test_df = time_based_split(df_feat)
    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

    # --- Severity ---
    print("\n=== Severity Models ===")
    y_sev_train = train_df["claim_amount"].values
    y_sev_test = test_df["claim_amount"].values
    sev_models, sev_results = train_severity_models(X_train, y_sev_train, X_test, y_sev_test)

    # --- Fraud ---
    print("\n=== Fraud Detection Models ===")
    y_fraud_train = train_df["is_fraud"].values
    y_fraud_test = test_df["is_fraud"].values
    fraud_models, fraud_results = train_fraud_models(X_train, y_fraud_train, X_test, y_fraud_test)

    # --- Pricing ---
    print("\n=== Risk Pricing Model ===")
    y_freq_train = train_df["prior_claims_count"].values
    premiums_test = test_df["premium"].values
    pricer, pricing_results = train_pricing_model(
        X_train, y_freq_train, y_sev_train, X_test, y_sev_test, premiums_test
    )

    # --- Save models ---
    print("\nSaving models...")
    for name, model in sev_models.items():
        joblib.dump(model, MODEL_DIR / f"severity_{name}.pkl")
    for name, model in fraud_models.items():
        joblib.dump(model, MODEL_DIR / f"fraud_{name}.pkl")
    joblib.dump(pricer, MODEL_DIR / "risk_pricer.pkl")
    joblib.dump(enc_maps, MODEL_DIR / "encoding_maps.pkl")

    # --- Save metrics ---
    all_metrics = {
        "severity": sev_results,
        "fraud": fraud_results,
        "pricing": pricing_results,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Models & metrics saved → {MODEL_DIR}/")

    # --- Save test data for evaluation ---
    test_df.to_csv(DATA_DIR / "test_set.csv", index=False)
    print("Done!")


if __name__ == "__main__":
    main()
