"""Feature engineering for insurance claims analytics."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures


SEVERITY_MAP = {"none": 0, "minor": 1, "moderate": 2, "severe": 3, "critical": 4}
DAMAGE_MAP = {"none": 0, "minor": 1, "moderate": 2, "major": 3, "total_loss": 4}

CATEGORICAL_COLS = [
    "gender",
    "marital_status",
    "policy_type",
    "region",
    "incident_type",
    "injury_severity",
    "property_damage_extent",
]


def add_loss_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Loss ratio = claim_amount / premium and related features."""
    df = df.copy()
    df["loss_ratio"] = (df["claim_amount"] / df["premium"].replace(0, np.nan)).fillna(0)
    df["loss_ratio_clipped"] = df["loss_ratio"].clip(0, 20)
    df["claim_to_coverage"] = df["claim_amount"] / df["coverage_amount"].replace(0, 1)
    df["premium_to_coverage"] = df["premium"] / df["coverage_amount"].replace(0, 1)
    df["deductible_ratio"] = df["deductible"] / df["coverage_amount"].replace(0, 1)
    df["net_claim"] = (df["claim_amount"] - df["deductible"]).clip(lower=0)
    return df


def add_claim_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Claim frequency features based on policyholder history."""
    df = df.copy()
    df["claims_per_year"] = df["prior_claims_count"] / df["years_as_customer"].replace(0, 1)
    df["is_frequent_claimant"] = (df["claims_per_year"] > 0.5).astype(int)
    df["high_prior_claims"] = (df["prior_claims_count"] > 3).astype(int)
    df["new_customer"] = (df["years_as_customer"] < 2).astype(int)
    df["loyal_customer"] = (df["years_as_customer"] >= 10).astype(int)
    return df


def add_risk_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Create risk segments based on key characteristics."""
    df = df.copy()

    # Age-based risk
    df["age_group"] = pd.cut(
        df["policyholder_age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["young", "young_adult", "middle", "senior", "elderly"],
    )

    # Credit risk tier
    df["credit_tier"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 800, 900],
        labels=["poor", "fair", "good", "very_good", "excellent"],
    )

    # Coverage tier
    coverage_pctiles = df["coverage_amount"].quantile([0.25, 0.5, 0.75])
    df["coverage_tier"] = pd.cut(
        df["coverage_amount"],
        bins=[0, coverage_pctiles[0.25], coverage_pctiles[0.5], coverage_pctiles[0.75], np.inf],
        labels=["low", "medium", "high", "very_high"],
    )

    # Overall risk score (heuristic)
    df["injury_severity_num"] = df["injury_severity"].map(SEVERITY_MAP).fillna(0)
    df["property_damage_num"] = df["property_damage_extent"].map(DAMAGE_MAP).fillna(0)
    df["risk_score"] = (
        0.2 * df["claims_per_year"].clip(0, 3) / 3
        + 0.15 * (1 - df["credit_score"].clip(300, 850).sub(300).div(550))
        + 0.15 * df["injury_severity_num"] / 4
        + 0.15 * df["property_damage_num"] / 4
        + 0.1 * df["new_customer"]
        + 0.1 * df["claim_to_coverage"].clip(0, 1)
        + 0.15 * (1 - df["police_report_filed"])
    )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction between key features."""
    df = df.copy()

    # Age × policy type (via one-hot interaction)
    for pt in ["auto", "home", "health", "life"]:
        df[f"age_x_{pt}"] = df["policyholder_age"] * (df["policy_type"] == pt).astype(int)

    # Credit × prior claims
    df["credit_x_claims"] = df["credit_score"] * df["prior_claims_count"]

    # Coverage × injury
    df["coverage_x_injury"] = df["coverage_amount"] * df["injury_severity_num"]

    # Witnesses × police report
    df["witnesses_x_report"] = df["witnesses"] * df["police_report_filed"]

    return df


def add_polynomial_features(df: pd.DataFrame, cols: list[str] | None = None, degree: int = 2) -> pd.DataFrame:
    """Add polynomial transforms for selected numeric columns."""
    df = df.copy()
    if cols is None:
        cols = ["policyholder_age", "credit_score", "prior_claims_count"]
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return df
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_arr = poly.fit_transform(df[existing].fillna(0))
    poly_names = poly.get_feature_names_out(existing)
    # only add the new columns (skip original)
    new_mask = [n not in existing for n in poly_names]
    new_names = [n.replace(" ", "_") for n, m in zip(poly_names, new_mask) if m]
    new_arr = poly_arr[:, new_mask]
    for name, col_data in zip(new_names, new_arr.T):
        df[f"poly_{name}"] = col_data
    return df


def add_target_encoding(
    df: pd.DataFrame,
    target_col: str,
    cat_cols: list[str] | None = None,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Target encoding for categorical columns with global-mean smoothing."""
    df = df.copy()
    if cat_cols is None:
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    global_mean = df[target_col].mean()
    encoding_maps: dict[str, dict] = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        stats = df.groupby(col)[target_col].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        encoding_maps[col] = smooth.to_dict()
        df[f"{col}_te"] = df[col].map(smooth).fillna(global_mean)

    return df, encoding_maps


def add_seasonal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract seasonal / temporal patterns from claim_date."""
    df = df.copy()
    dt = pd.to_datetime(df["claim_date"])
    df["claim_month"] = dt.dt.month
    df["claim_quarter"] = dt.dt.quarter
    df["claim_day_of_week"] = dt.dt.dayofweek
    df["claim_year"] = dt.dt.year
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["claim_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["claim_month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["claim_day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["claim_day_of_week"] / 7)

    # Known high-claim seasons (winter weather, summer travel)
    df["high_claim_season"] = df["claim_month"].isin([1, 2, 6, 7, 12]).astype(int)
    return df


def engineer_features(df: pd.DataFrame, target_col: str = "claim_amount") -> tuple[pd.DataFrame, dict]:
    """Run full feature engineering pipeline."""
    df = add_loss_ratio_features(df)
    df = add_claim_frequency_features(df)
    df = add_risk_segmentation(df)
    df = add_interaction_features(df)
    df = add_polynomial_features(df)
    df = add_seasonal_patterns(df)
    df, encoding_maps = add_target_encoding(df, target_col=target_col)
    return df, encoding_maps


def main():
    raw_path = Path("data/raw/claims.csv")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path, parse_dates=["claim_date"])
    df_feat, enc_maps = engineer_features(df)
    df_feat.to_csv(out_dir / "claims_featured.csv", index=False)
    print(f"Engineered {df_feat.shape[1]} features → {out_dir / 'claims_featured.csv'}")
    print(f"New columns: {df_feat.shape[1] - df.shape[1]}")


if __name__ == "__main__":
    main()
