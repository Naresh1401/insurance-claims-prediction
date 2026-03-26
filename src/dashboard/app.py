"""Streamlit dashboard for Insurance Claims Analytics."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODEL_DIR = Path("data/models")
DATA_DIR = Path("data/processed")
PLOT_DIR = Path("plots")

st.set_page_config(page_title="Insurance Claims Analytics", page_icon="🏥", layout="wide")


# ── Data loading (cached) ────────────────────────────────────────────


@st.cache_data
def load_data():
    featured = DATA_DIR / "claims_featured.csv"
    if featured.exists():
        return pd.read_csv(featured)
    raw = Path("data/raw/claims.csv")
    if raw.exists():
        return pd.read_csv(raw)
    return None


@st.cache_resource
def load_model(name: str):
    path = MODEL_DIR / f"{name}.pkl"
    if path.exists():
        return joblib.load(path)
    return None


def load_metrics():
    path = MODEL_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Sidebar navigation ──────────────────────────────────────────────

st.sidebar.title("🏥 Insurance Claims Analytics")
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Claims Overview",
        "🔍 Individual Claim Assessment",
        "🚨 Fraud Investigation",
        "💰 Pricing Simulator",
        "📈 Portfolio Risk Dashboard",
    ],
)


# ── Page: Claims Overview ────────────────────────────────────────────

if page == "📊 Claims Overview":
    st.title("📊 Claims Analysis Overview")
    df = load_data()
    if df is None:
        st.warning("No data found. Run `make data` and `make train` first.")
        st.stop()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Claims", f"{len(df):,}")
    col2.metric("Avg Claim", f"${df['claim_amount'].mean():,.0f}")
    col3.metric("Fraud Rate", f"{df['is_fraud'].mean():.1%}")
    col4.metric("Avg Premium", f"${df['premium'].mean():,.0f}")

    st.subheader("Claim Amount Distribution")
    fig = px.histogram(
        df, x="claim_amount", nbins=80, color="policy_type",
        title="Claim Amount by Policy Type", marginal="box",
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Claims by Region")
        fig = px.pie(df, names="region", title="Claims Distribution by Region")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Severity by Injury Level")
        fig = px.box(
            df, x="injury_severity", y="claim_amount",
            title="Claim Amount by Injury Severity",
            category_orders={"injury_severity": ["none", "minor", "moderate", "severe", "critical"]},
        )
        st.plotly_chart(fig, use_container_width=True)

    if "claim_date" in df.columns:
        st.subheader("Claims Over Time")
        df["claim_date"] = pd.to_datetime(df["claim_date"])
        monthly = df.set_index("claim_date").resample("M")["claim_amount"].agg(["sum", "count"]).reset_index()
        monthly.columns = ["date", "total_amount", "count"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["date"], y=monthly["count"], name="# Claims"))
        fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["total_amount"], name="Total $", yaxis="y2"))
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", title="Total Amount ($)"),
            yaxis=dict(title="Claim Count"), title="Monthly Claims Trend",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Individual Claim Assessment ────────────────────────────────

elif page == "🔍 Individual Claim Assessment":
    st.title("🔍 Individual Claim Assessment")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Policyholder Age", 18, 100, 42)
        gender = st.selectbox("Gender", ["M", "F"])
        policy_type = st.selectbox("Policy Type", ["auto", "home", "health", "life"])
    with col2:
        coverage = st.number_input("Coverage Amount ($)", 1000, 5_000_000, 50_000)
        premium = st.number_input("Premium ($)", 100, 50_000, 1_800)
        deductible = st.number_input("Deductible ($)", 0, 10_000, 500)
    with col3:
        years = st.number_input("Years as Customer", 0, 50, 5)
        prior_claims = st.number_input("Prior Claims Count", 0, 20, 1)
        credit = st.slider("Credit Score", 300, 850, 720)

    col4, col5 = st.columns(2)
    with col4:
        police = st.selectbox("Police Report Filed", [1, 0])
        witnesses = st.number_input("Witnesses", 0, 10, 2)
    with col5:
        injury = st.select_slider("Injury Severity", options=[0, 1, 2, 3, 4], value=1)
        damage = st.select_slider("Property Damage", options=[0, 1, 2, 3, 4], value=2)

    if st.button("Assess Claim", type="primary"):
        sev_model = load_model("severity_xgboost")
        fraud_model = load_model("fraud_xgboost_smote")
        pricer = load_model("risk_pricer")

        if not all([sev_model, fraud_model, pricer]):
            st.error("Models not found. Run training first.")
            st.stop()

        is_auto = int(policy_type == "auto")
        is_home = int(policy_type == "home")
        is_health = int(policy_type == "health")
        is_life = int(policy_type == "life")
        claims_per_yr = prior_claims / max(years, 1)

        vec = np.array([[
            age, coverage, premium, deductible, years, prior_claims, credit,
            police, witnesses, injury, damage,
            min((premium / coverage) * 20, 20), 0, premium / coverage,
            deductible / coverage, claims_per_yr, int(claims_per_yr > 0.5),
            int(prior_claims > 3), int(years < 2), int(years >= 10), 0.5,
            age * is_auto, age * is_home, age * is_health, age * is_life,
            credit * prior_claims, coverage * injury, witnesses * police,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]], dtype=np.float32)

        severity = float(sev_model.predict(vec)[0])
        fraud_prob = float(fraud_model.predict_proba(vec)[0, 1])
        pure_prem = float(pricer.predict_pure_premium(vec)[0])
        gross_prem = float(pricer.predict_gross_premium(vec)[0])

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Severity", f"${severity:,.2f}")
        r2.metric("Fraud Probability", f"{fraud_prob:.1%}")
        r3.metric("Recommended Premium", f"${gross_prem:,.2f}")

        if fraud_prob > 0.6:
            st.error("⚠️ HIGH FRAUD RISK — Recommend manual investigation.")
        elif fraud_prob > 0.3:
            st.warning("⚠️ MODERATE FRAUD RISK — Additional verification advised.")
        else:
            st.success("✅ LOW FRAUD RISK — Claim appears legitimate.")


# ── Page: Fraud Investigation ────────────────────────────────────────

elif page == "🚨 Fraud Investigation":
    st.title("🚨 Fraud Investigation View")
    df = load_data()
    if df is None:
        st.warning("No data found.")
        st.stop()

    metrics = load_metrics()
    if "fraud" in metrics:
        st.subheader("Model Performance")
        perf_df = pd.DataFrame(metrics["fraud"]).T
        st.dataframe(perf_df.style.highlight_max(axis=0), use_container_width=True)

    # Fraud analysis
    fraud_df = df[df["is_fraud"] == 1]
    legit_df = df[df["is_fraud"] == 0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Fraudulent", f"{len(fraud_df):,}")
    col2.metric("Avg Fraud Claim", f"${fraud_df['claim_amount'].mean():,.0f}")
    col3.metric("Fraud Uplift", f"{fraud_df['claim_amount'].mean() / legit_df['claim_amount'].mean():.2f}x")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(
            df, x="claim_amount", color="is_fraud", nbins=60, barmode="overlay",
            title="Claim Distribution: Fraud vs Legitimate", opacity=0.6,
            color_discrete_map={0: "steelblue", 1: "crimson"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fraud_by_type = df.groupby("policy_type")["is_fraud"].mean().reset_index()
        fig = px.bar(fraud_by_type, x="policy_type", y="is_fraud", title="Fraud Rate by Policy Type")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    # Saved plots
    for plot_name in ["fraud_roc_pr_curves.png", "fraud_confusion_matrices.png", "fraud_cost_benefit.png"]:
        path = PLOT_DIR / plot_name
        if path.exists():
            st.image(str(path), caption=plot_name.replace("_", " ").replace(".png", "").title())


# ── Page: Pricing Simulator ──────────────────────────────────────────

elif page == "💰 Pricing Simulator":
    st.title("💰 Pricing Simulator")

    st.subheader("Adjust Risk Factors")
    col1, col2 = st.columns(2)
    with col1:
        sim_age = st.slider("Age", 18, 85, 40)
        sim_credit = st.slider("Credit Score", 300, 850, 700)
        sim_policy = st.selectbox("Policy Type", ["auto", "home", "health", "life"], key="sim_pol")
    with col2:
        sim_coverage = st.number_input("Coverage ($)", 5000, 2_000_000, 100_000, key="sim_cov")
        sim_prior = st.slider("Prior Claims", 0, 10, 1)
        sim_years = st.slider("Years as Customer", 0, 30, 5)

    pricer = load_model("risk_pricer")
    if pricer and st.button("Calculate Premium", type="primary"):
        is_auto = int(sim_policy == "auto")
        is_home = int(sim_policy == "home")
        is_health = int(sim_policy == "health")
        is_life = int(sim_policy == "life")
        cpy = sim_prior / max(sim_years, 1)

        vec = np.array([[
            sim_age, sim_coverage, 0, 500, sim_years, sim_prior, sim_credit,
            1, 1, 1, 1, 0, 0, 0, 500 / sim_coverage, cpy,
            int(cpy > 0.5), int(sim_prior > 3), int(sim_years < 2),
            int(sim_years >= 10), 0.5,
            sim_age * is_auto, sim_age * is_home, sim_age * is_health,
            sim_age * is_life, sim_credit * sim_prior, sim_coverage, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]], dtype=np.float32)

        pure = float(pricer.predict_pure_premium(vec)[0])
        gross = float(pricer.predict_gross_premium(vec)[0])

        c1, c2, c3 = st.columns(3)
        c1.metric("Pure Premium", f"${max(pure, 0):,.2f}")
        c2.metric("Gross Premium", f"${max(gross, 0):,.2f}")
        c3.metric("Loading Factor", f"{gross / max(pure, 1):.2f}x")

        # Sensitivity analysis
        st.subheader("Sensitivity: Premium vs Age")
        ages = list(range(18, 86, 2))
        premiums = []
        for a in ages:
            v = vec.copy()
            v[0, 0] = a
            v[0, 21] = a * is_auto
            v[0, 22] = a * is_home
            v[0, 23] = a * is_health
            v[0, 24] = a * is_life
            premiums.append(float(pricer.predict_gross_premium(v)[0]))
        fig = px.line(x=ages, y=premiums, labels={"x": "Age", "y": "Gross Premium ($)"})
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Portfolio Risk Dashboard ───────────────────────────────────

elif page == "📈 Portfolio Risk Dashboard":
    st.title("📈 Portfolio Risk Dashboard")
    df = load_data()
    metrics = load_metrics()
    if df is None:
        st.warning("No data available.")
        st.stop()

    # Portfolio summary
    st.subheader("Portfolio Summary")
    total_premium = df["premium"].sum()
    total_claims = df["claim_amount"].sum()
    overall_lr = total_claims / total_premium

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Premium", f"${total_premium:,.0f}")
    c2.metric("Total Claims", f"${total_claims:,.0f}")
    c3.metric("Loss Ratio", f"{overall_lr:.2%}")
    c4.metric("Combined Ratio", f"{overall_lr + 0.15:.2%}")

    # Loss ratio by policy type
    st.subheader("Loss Ratio by Policy Type")
    lr_by_type = df.groupby("policy_type").agg(
        total_premium=("premium", "sum"),
        total_claims=("claim_amount", "sum"),
        count=("policy_id" if "policy_id" in df.columns else "policyholder_age", "size"),
    )
    lr_by_type["loss_ratio"] = lr_by_type["total_claims"] / lr_by_type["total_premium"]
    fig = px.bar(
        lr_by_type.reset_index(), x="policy_type", y="loss_ratio",
        color="loss_ratio", color_continuous_scale="RdYlGn_r",
        title="Loss Ratio by Policy Type",
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even")
    st.plotly_chart(fig, use_container_width=True)

    # Risk heatmap
    if "risk_score" in df.columns:
        st.subheader("Risk Heatmap: Age Group × Region")
        df["age_bin"] = pd.cut(df["policyholder_age"], bins=[0, 25, 35, 50, 65, 100],
                               labels=["18-25", "26-35", "36-50", "51-65", "66+"])
        pivot = df.pivot_table(values="risk_score", index="age_bin", columns="region", aggfunc="mean")
        fig = px.imshow(pivot, color_continuous_scale="YlOrRd", title="Average Risk Score")
        st.plotly_chart(fig, use_container_width=True)

    # Model metrics summary
    if metrics:
        st.subheader("Model Performance Summary")
        for section, vals in metrics.items():
            st.write(f"**{section.title()}**")
            if isinstance(vals, dict) and all(isinstance(v, dict) for v in vals.values()):
                st.dataframe(pd.DataFrame(vals).T, use_container_width=True)
            else:
                st.json(vals)

    # Saved evaluation plots
    st.subheader("Evaluation Plots")
    for plot_file in sorted(PLOT_DIR.glob("*.png")) if PLOT_DIR.exists() else []:
        st.image(str(plot_file), caption=plot_file.stem.replace("_", " ").title())
