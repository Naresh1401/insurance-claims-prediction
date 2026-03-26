"""Model evaluation: plots and analysis for severity, fraud, and pricing."""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")

MODEL_DIR = Path("data/models")
DATA_DIR = Path("data/processed")
PLOT_DIR = Path("plots")

FEATURE_COLS = [
    "policyholder_age", "coverage_amount", "premium", "deductible",
    "years_as_customer", "prior_claims_count", "credit_score",
    "police_report_filed", "witnesses", "injury_severity_num",
    "property_damage_num", "loss_ratio_clipped", "claim_to_coverage",
    "premium_to_coverage", "deductible_ratio", "claims_per_year",
    "is_frequent_claimant", "high_prior_claims", "new_customer",
    "loyal_customer", "risk_score", "age_x_auto", "age_x_home",
    "age_x_health", "age_x_life", "credit_x_claims", "coverage_x_injury",
    "witnesses_x_report", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "high_claim_season", "is_weekend",
    "gender_te", "marital_status_te", "policy_type_te", "region_te",
    "incident_type_te", "injury_severity_te", "property_damage_extent_te",
]


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].fillna(0).values.astype(np.float32)


# ── Severity evaluation ──────────────────────────────────────────────


def plot_actual_vs_predicted(y_true, y_pred, model_name: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sample_idx = np.random.RandomState(42).choice(len(y_true), min(5000, len(y_true)), replace=False)
    ax.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, s=10)
    lims = [0, max(y_true[sample_idx].max(), y_pred[sample_idx].max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Claim Amount ($)")
    ax.set_ylabel("Predicted Claim Amount ($)")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"severity_actual_vs_pred_{model_name}.png", dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name: str):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual distribution
    axes[0].hist(residuals, bins=80, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title(f"Residual Distribution — {model_name}")
    axes[0].set_xlabel("Residual ($)")

    # Residuals vs predicted
    sample_idx = np.random.RandomState(42).choice(len(y_true), min(5000, len(y_true)), replace=False)
    axes[1].scatter(y_pred[sample_idx], residuals[sample_idx], alpha=0.3, s=10)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted ($)")
    axes[1].set_ylabel("Residual ($)")
    axes[1].set_title(f"Residuals vs Predicted — {model_name}")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"severity_residuals_{model_name}.png", dpi=150)
    plt.close(fig)


def plot_qq(y_true, y_pred, model_name: str):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot — {model_name}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"severity_qq_{model_name}.png", dpi=150)
    plt.close(fig)


def evaluate_severity(test_df, X_test):
    print("=== Severity Evaluation ===")
    y_test = test_df["claim_amount"].values
    sev_models = {}
    for p in MODEL_DIR.glob("severity_*.pkl"):
        name = p.stem.replace("severity_", "")
        sev_models[name] = joblib.load(p)

    for name, model in sev_models.items():
        preds = model.predict(X_test)
        print(f"  {name}: MAE={mean_absolute_error(y_test, preds):.2f}")
        plot_actual_vs_predicted(y_test, preds, name)
        plot_residuals(y_test, preds, name)
        plot_qq(y_test, preds, name)


# ── Fraud evaluation ─────────────────────────────────────────────────


def plot_roc_pr(y_true, models_proba: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curves
    for name, proba in models_proba.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[0].plot(fpr, tpr, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend()

    # Precision-Recall curves
    for name, proba in models_proba.items():
        prec, rec, _ = precision_recall_curve(y_true, proba)
        axes[1].plot(rec, prec, label=name)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fraud_roc_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(y_true, models_preds: dict):
    n = len(models_preds)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, preds) in zip(axes, models_preds.items()):
        cm = confusion_matrix(y_true, preds)
        ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fraud_confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_cost_benefit(y_true, proba, avg_fraud_cost: float = 15000, investigation_cost: float = 500):
    thresholds = np.linspace(0.01, 0.99, 100)
    net_savings = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        savings = tp * avg_fraud_cost - fp * investigation_cost - fn * avg_fraud_cost
        net_savings.append(savings)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, net_savings)
    best_idx = np.argmax(net_savings)
    ax.axvline(thresholds[best_idx], color="red", linestyle="--", label=f"Optimal: {thresholds[best_idx]:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net Savings ($)")
    ax.set_title("Cost-Benefit Analysis of Fraud Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fraud_cost_benefit.png", dpi=150)
    plt.close(fig)


def evaluate_fraud(test_df, X_test):
    print("\n=== Fraud Evaluation ===")
    y_test = test_df["is_fraud"].values
    fraud_models = {}
    for p in MODEL_DIR.glob("fraud_*.pkl"):
        name = p.stem.replace("fraud_", "")
        fraud_models[name] = joblib.load(p)

    models_proba = {}
    models_preds = {}
    for name, model in fraud_models.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        models_proba[name] = proba
        models_preds[name] = preds

    plot_roc_pr(y_test, models_proba)
    plot_confusion_matrices(y_test, models_preds)

    # Cost-benefit with best model (xgboost_smote if available)
    best_name = "xgboost_smote" if "xgboost_smote" in models_proba else list(models_proba.keys())[0]
    plot_cost_benefit(y_test, models_proba[best_name])


# ── Pricing evaluation ───────────────────────────────────────────────


def plot_loss_ratio_by_segment(test_df, X_test, pricer):
    segments = test_df["policy_type"].values if "policy_type" in test_df.columns else np.full(len(test_df), "all")
    lr_df = pricer.loss_ratio_by_segment(
        X_test, test_df["claim_amount"].values, test_df["premium"].values, segments
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(lr_df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], lr_df["actual_loss_ratio"], width, label="Actual", color="steelblue")
    ax.bar([i + width / 2 for i in x], lr_df["predicted_loss_ratio"], width, label="Predicted", color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels(lr_df.index, rotation=45)
    ax.set_ylabel("Loss Ratio")
    ax.set_title("Loss Ratio by Policy Type")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Break-even")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pricing_loss_ratio.png", dpi=150)
    plt.close(fig)


def plot_lorenz_curve(y_true, y_pred):
    """Lorenz curve for pricing model (ordered by predicted risk)."""
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    cum_claims = np.cumsum(y_sorted) / y_sorted.sum()
    cum_pct = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cum_pct, cum_claims, label="Model", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("Cumulative % of Policies (ordered by predicted risk)")
    ax.set_ylabel("Cumulative % of Claims")
    ax.set_title("Lorenz Curve — Risk Pricing Model")
    ax.legend()
    ax.fill_between(cum_pct, cum_claims, cum_pct, alpha=0.15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pricing_lorenz_curve.png", dpi=150)
    plt.close(fig)


def plot_lift_chart(y_true, y_pred, n_bins: int = 10):
    """Lift chart: actual/expected ratio by predicted risk decile."""
    df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    df["decile"] = pd.qcut(df["predicted"], n_bins, labels=False, duplicates="drop")
    agg = df.groupby("decile").agg(
        avg_actual=("actual", "mean"), avg_predicted=("predicted", "mean"), count=("actual", "size")
    )
    overall_mean = y_true.mean()
    agg["lift"] = agg["avg_actual"] / overall_mean

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(agg.index, agg["lift"], color="teal", edgecolor="black")
    ax.axhline(1.0, color="red", linestyle="--", label="Baseline")
    ax.set_xlabel("Risk Decile (0=lowest)")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Chart — Risk Pricing Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pricing_lift_chart.png", dpi=150)
    plt.close(fig)


def evaluate_pricing(test_df, X_test):
    print("\n=== Pricing Evaluation ===")
    pricer = joblib.load(MODEL_DIR / "risk_pricer.pkl")
    predicted = pricer.predict_gross_premium(X_test)
    y_sev = test_df["claim_amount"].values

    plot_loss_ratio_by_segment(test_df, X_test, pricer)
    plot_lorenz_curve(y_sev, predicted)
    plot_lift_chart(y_sev, predicted)

    adequacy = pricer.premium_adequacy(X_test, test_df["premium"].values)
    print(f"  Adequacy ratio: {adequacy['adequacy_ratio']:.4f}")
    print(f"  Underpriced: {adequacy['underprice_pct']:.2%}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(DATA_DIR / "test_set.csv")
    X_test = prepare_features(test_df)

    evaluate_severity(test_df, X_test)
    evaluate_fraud(test_df, X_test)
    evaluate_pricing(test_df, X_test)

    print(f"\nAll plots saved → {PLOT_DIR}/")


if __name__ == "__main__":
    main()
