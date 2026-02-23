# SMS Spam (Threat Detection) Demo

This repo follows a 10-day ML engineering crash course plan to build an end-to-end SMS spam/phishing (“threat”) detection + triage service.

## Problem Statement

**Objective**: Detect spam/phishing messages ("threats") in SMS text to enable automated triage and analyst review.

**What constitutes a threat**: 
- Spam messages (promotional, prize/lottery scams, premium rate services)
- Phishing attempts (credential harvesting, account verification requests)
- Messages with commercial intent targeting personal mobile users

**Business impact**:
- **False Positives**: Legitimate messages blocked → user frustration, missed communications
- **False Negatives**: Spam/phishing delivered → security risk, user annoyance, potential fraud

## Performance Metrics & Targets

**Primary Metrics**: Precision and Recall (security-focused evaluation)
- **Precision**: Of messages flagged as threats, how many are actually threats?
- **Recall**: Of actual threats, how many do we catch?

**Current Performance** (TF-IDF + Logistic Regression baseline):
- **Test Set**: Precision=100%, Recall=89%, F1=0.942, ROC-AUC=0.993
- **Validation Set**: Precision=100%, Recall=80%, F1=0.889, ROC-AUC=0.997

**Target**: Maintain Recall ≥ 85% while keeping Precision ≥ 95% (adjustable based on triage capacity)

## Model Selection

**Baseline**: TF-IDF + Logistic Regression
- **Rationale**: Outperformed Random Forest (F1: 0.942 vs 0.892), faster inference, interpretable
- **Architecture**: sklearn Pipeline with TfidfVectorizer → LogisticRegression
- **Features**: Unigrams + bigrams, max 50k features, L2 regularization (C=4.0)

## Day 0-1 Status

- **AWS**: CLI configured, ECR repo created, Docker working
- **Data**: processed CSV generated (5,572 messages: 87% ham, 13% spam)
- **Splits**: leakage-safe train/val/test splits created (grouped by exact text)
- **Baseline**: TF-IDF + Logistic Regression trained and evaluated

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data processing

Generate `data/processed/sms_spam.csv` from the raw dataset:

```bash
python scripts/process_sms_spam.py
```

## Train/val/test splits

Create splits under `data/splits/`:

```bash
python scripts/make_splits.py
```

## Baseline model

Train a TF-IDF + Logistic Regression baseline and write artifacts:

```bash
python training/train.py
```

Outputs:
- `artifacts/model.joblib`
- `reports/metrics.json`
