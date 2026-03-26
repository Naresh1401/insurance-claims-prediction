# 🏥 Insurance Claims Severity & Fraud Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ML-powered insurance claims analytics platform** for severity prediction, fraud detection, risk-based pricing, and loss ratio optimization across auto, home, health, and life insurance lines.

**Author:** Naresh Sampangi

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Insurance Analytics Platform                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌───────────────┐   ┌────────────────────────┐ │
│  │ Raw      │──▶│ Feature       │──▶│ Model Pipeline         │ │
│  │ Claims   │   │ Engineering   │   │                        │ │
│  │ Data     │   │               │   │  ┌──────────────────┐  │ │
│  │ (50K)    │   │ • Loss Ratio  │   │  │ Severity Model   │  │ │
│  └──────────┘   │ • Risk Seg.   │   │  │ (Tweedie/XGB)    │  │ │
│                 │ • Interactions │   │  ├──────────────────┤  │ │
│                 │ • Seasonality  │   │  │ Fraud Detector   │  │ │
│                 │ • Target Enc.  │   │  │ (IsoForest/SMOTE)│  │ │
│                 └───────────────┘   │  ├──────────────────┤  │ │
│                                     │  │ Risk Pricer      │  │ │
│                                     │  │ (GLM Freq×Sev)   │  │ │
│                                     │  └──────────────────┘  │ │
│                                     └────────────────────────┘ │
│                                              │                  │
│                    ┌─────────────────────────┼──────────┐      │
│                    │                         │          │      │
│               ┌────▼─────┐          ┌───────▼───┐  ┌───▼───┐  │
│               │ FastAPI   │          │ Streamlit │  │ Saved │  │
│               │ Endpoints │          │ Dashboard │  │ Models│  │
│               │           │          │           │  │ (.pkl)│  │
│               │ /severity │          │ • Claims  │  └───────┘  │
│               │ /fraud    │          │ • Fraud   │             │
│               │ /premium  │          │ • Pricing │             │
│               └───────────┘          │ • Risk    │             │
│                                      └───────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Results

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| XGBoost (Tweedie) | Severity Prediction | MAE | ~$2,800 |
| LightGBM | Severity Prediction | RMSE | ~$5,200 |
| XGBoost + SMOTE | Fraud Detection | AUC-ROC | ~0.94 |
| Isolation Forest | Fraud Detection | Precision@10% | ~0.72 |
| GLM (Poisson+Gamma) | Risk Pricing | Gini Coefficient | ~0.38 |
| LightGBM | Fraud Detection | F1-Score | ~0.61 |

## Project Structure

```
insurance-claims-prediction/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
└── src/
    ├── data/
    │   └── generate_data.py
    ├── features/
    │   └── feature_engineering.py
    ├── models/
    │   ├── severity_model.py
    │   ├── fraud_detector.py
    │   ├── risk_pricer.py
    │   ├── train.py
    │   └── evaluate.py
    ├── api/
    │   └── app.py
    └── dashboard/
        └── app.py
```

## Quick Start

```bash
# Install dependencies
make install

# Generate synthetic data
make data

# Train all models
make train

# Evaluate models
make evaluate

# Start API server
make api

# Launch dashboard
make dashboard
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/severity` | POST | Predict claim severity amount |
| `/detect/fraud` | POST | Detect fraudulent claims |
| `/price/premium` | POST | Calculate risk-based premium |
| `/health` | GET | Health check |

## Key Features

- **Multi-line coverage**: Auto, Home, Health, Life insurance
- **Severity modeling**: Gamma/Tweedie GLMs handle zero-inflated claim distributions
- **Fraud detection**: Ensemble of supervised (XGBoost+SMOTE) and unsupervised (Isolation Forest) methods
- **Risk pricing**: Actuarial frequency × severity framework with GLM backbone
- **Explainability**: SHAP values for model interpretability
- **Production-ready**: FastAPI serving, Docker deployment, Streamlit monitoring

## License

MIT License
