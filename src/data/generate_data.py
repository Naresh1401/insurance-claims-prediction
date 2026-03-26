"""Generate synthetic insurance claims data with realistic distributions."""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_CLAIMS = 50_000

POLICY_TYPES = ["auto", "home", "health", "life"]
GENDERS = ["M", "F"]
MARITAL_STATUSES = ["single", "married", "divorced", "widowed"]
REGIONS = ["northeast", "southeast", "midwest", "southwest", "west"]
AUTO_INCIDENT_TYPES = ["collision", "theft", "vandalism", "weather"]
HOME_INCIDENT_TYPES = ["fire", "water_damage", "theft", "weather", "structural"]
HEALTH_INCIDENT_TYPES = ["emergency", "surgery", "chronic", "preventive"]
LIFE_INCIDENT_TYPES = ["death", "critical_illness", "disability"]
INJURY_SEVERITIES = ["none", "minor", "moderate", "severe", "critical"]
PROPERTY_DAMAGE_EXTENTS = ["none", "minor", "moderate", "major", "total_loss"]


def generate_claims(n: int = N_CLAIMS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    # --- Base policyholder features ---
    policy_ids = [f"POL-{i:06d}" for i in range(n)]
    start = pd.Timestamp("2019-01-01")
    claim_dates = start + pd.to_timedelta(rng.randint(0, 365 * 5, n), unit="D")
    ages = np.clip(rng.normal(45, 14, n).astype(int), 18, 85)
    genders = rng.choice(GENDERS, n, p=[0.52, 0.48])
    marital = rng.choice(MARITAL_STATUSES, n, p=[0.28, 0.48, 0.18, 0.06])
    policy_type = rng.choice(POLICY_TYPES, n, p=[0.40, 0.25, 0.25, 0.10])
    years_as_customer = np.clip(rng.exponential(7, n).astype(int), 0, 40)
    prior_claims = rng.poisson(1.2, n)
    credit_scores = np.clip(rng.normal(700, 80, n).astype(int), 300, 850)
    regions = rng.choice(REGIONS, n, p=[0.20, 0.22, 0.20, 0.18, 0.20])

    # --- Policy financials ---
    coverage = np.where(
        policy_type == "auto",
        rng.lognormal(10.2, 0.4, n),
        np.where(
            policy_type == "home",
            rng.lognormal(12.0, 0.5, n),
            np.where(
                policy_type == "health",
                rng.lognormal(10.8, 0.3, n),
                rng.lognormal(11.5, 0.6, n),
            ),
        ),
    ).round(2)

    premium = (coverage * rng.uniform(0.01, 0.06, n)).round(2)
    deductible = np.where(
        policy_type == "auto",
        rng.choice([250, 500, 1000, 2000], n, p=[0.2, 0.4, 0.3, 0.1]),
        np.where(
            policy_type == "home",
            rng.choice([500, 1000, 2500, 5000], n, p=[0.15, 0.35, 0.35, 0.15]),
            np.where(
                policy_type == "health",
                rng.choice([500, 1000, 2000, 5000], n, p=[0.2, 0.3, 0.3, 0.2]),
                rng.choice([0, 500, 1000], n, p=[0.5, 0.3, 0.2]),
            ),
        ),
    ).astype(float)

    # --- Policy-type-specific features ---
    vehicle_age = np.where(policy_type == "auto", rng.poisson(5, n), np.nan)
    property_age = np.where(policy_type == "home", rng.poisson(25, n), np.nan)
    bmi = np.where(
        policy_type == "health", np.clip(rng.normal(27, 5, n), 16, 50).round(1), np.nan
    )
    smoker = np.where(
        policy_type == "health", rng.choice([0, 1], n, p=[0.78, 0.22]), np.nan
    )

    # --- Incident features ---
    incident_type = np.empty(n, dtype=object)
    for i in range(n):
        pt = policy_type[i]
        if pt == "auto":
            incident_type[i] = rng.choice(AUTO_INCIDENT_TYPES)
        elif pt == "home":
            incident_type[i] = rng.choice(HOME_INCIDENT_TYPES)
        elif pt == "health":
            incident_type[i] = rng.choice(HEALTH_INCIDENT_TYPES)
        else:
            incident_type[i] = rng.choice(LIFE_INCIDENT_TYPES)

    police_report = rng.choice([0, 1], n, p=[0.45, 0.55])
    witnesses = rng.poisson(1.5, n)
    injury_severity = rng.choice(
        INJURY_SEVERITIES, n, p=[0.30, 0.30, 0.22, 0.13, 0.05]
    )
    property_damage = rng.choice(
        PROPERTY_DAMAGE_EXTENTS, n, p=[0.20, 0.30, 0.28, 0.15, 0.07]
    )

    # --- Target: claim_amount (regression) ---
    severity_map = {"none": 0, "minor": 1, "moderate": 2, "severe": 3, "critical": 4}
    damage_map = {"none": 0, "minor": 1, "moderate": 2, "major": 3, "total_loss": 4}
    injury_num = np.array([severity_map[s] for s in injury_severity])
    damage_num = np.array([damage_map[d] for d in property_damage])

    base_amount = np.where(
        policy_type == "auto",
        rng.gamma(2.5, 2000, n),
        np.where(
            policy_type == "home",
            rng.gamma(2.0, 5000, n),
            np.where(
                policy_type == "health",
                rng.gamma(3.0, 3000, n),
                rng.gamma(1.5, 15000, n),
            ),
        ),
    )

    claim_amount = (
        base_amount
        * (1 + 0.3 * injury_num + 0.25 * damage_num)
        * (1 + 0.005 * (ages - 30))
        * (1 - 0.01 * years_as_customer)
    )
    claim_amount = np.clip(claim_amount, 100, coverage).round(2)

    # --- Target: is_fraud (binary, ~5%) ---
    fraud_logit = (
        -3.5
        + 0.4 * (prior_claims > 3).astype(float)
        + 0.3 * (years_as_customer < 2).astype(float)
        + 0.5 * (police_report == 0).astype(float)
        + 0.3 * (witnesses == 0).astype(float)
        + 0.002 * (claim_amount / 1000)
        - 0.002 * (credit_scores - 600)
        + 0.2 * (injury_num >= 3).astype(float)
        + rng.normal(0, 0.3, n)
    )
    fraud_prob = 1 / (1 + np.exp(-fraud_logit))
    is_fraud = (rng.random(n) < fraud_prob).astype(int)

    # inflate claim amounts for fraudulent claims
    claim_amount = np.where(
        is_fraud == 1, claim_amount * rng.uniform(1.3, 2.5, n), claim_amount
    ).round(2)
    claim_amount = np.clip(claim_amount, 100, coverage).round(2)

    df = pd.DataFrame(
        {
            "policy_id": policy_ids,
            "claim_date": claim_dates,
            "policyholder_age": ages,
            "gender": genders,
            "marital_status": marital,
            "policy_type": policy_type,
            "coverage_amount": coverage,
            "premium": premium,
            "deductible": deductible,
            "years_as_customer": years_as_customer,
            "prior_claims_count": prior_claims,
            "credit_score": credit_scores,
            "region": regions,
            "vehicle_age": vehicle_age,
            "property_age": property_age,
            "bmi": bmi,
            "smoker": smoker,
            "incident_type": incident_type,
            "police_report_filed": police_report,
            "witnesses": witnesses,
            "injury_severity": injury_severity,
            "property_damage_extent": property_damage,
            "claim_amount": claim_amount,
            "is_fraud": is_fraud,
        }
    )
    return df


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_claims()
    df.to_csv(out_dir / "claims.csv", index=False)
    print(f"Generated {len(df):,} claims → {out_dir / 'claims.csv'}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"Claim amount stats:\n{df['claim_amount'].describe().round(2)}")


if __name__ == "__main__":
    main()
