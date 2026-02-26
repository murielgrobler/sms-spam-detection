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

**Enhanced Model**: TF-IDF + Heuristic Features + Random Forest
- **Performance**: Test F1=0.928, Recall=86.6% (slightly lower than baseline)
- **Architecture**: Custom pipeline combining TF-IDF with 19 engineered features
- **Status**: Baseline remains superior for this dataset

## Feature Engineering

**Heuristic Features** (19 total):
- **Link Detection**: URL patterns, link keywords (`link_present`, `link_keyword`, `link_any`)
- **Financial Content**: Money amounts, financial keywords, prize references (`money_amount`, `financial_keyword`, `prize_keyword`)
- **Urgency Indicators**: Time pressure, urgency keywords, exclamation marks (`urgency_keyword`, `time_pressure`, `multiple_exclamations`)
- **Account Security**: Phishing patterns, security keywords, institution mentions (`account_action`, `phishing_pattern`, `institution_mention`)
- **Meta Features**: Text length, caps usage, phone numbers (`text_length`, `all_caps_words`, `phone_number`)

**Key Insight**: While heuristic features are interpretable and production-ready, they didn't significantly improve performance over the strong TF-IDF baseline on this dataset.

## Project Status

✅ **Production Deployment**: Live SMS Spam Detection API on AWS ECS Fargate  
✅ **Public Endpoint**: https://sm-c8ad3b8c79064fe3a03635593d8d56a5.ecs.us-east-2.on.aws  
✅ **Containerization**: Docker image with production-ready FastAPI service  
✅ **Cloud Infrastructure**: ECS cluster with auto-scaling and load balancing  
✅ **Model Integration**: Enhanced model with heuristic features deployed  
✅ **Health Monitoring**: Comprehensive health checks and logging  

**Development Progress**:
- **Data**: processed CSV generated (5,572 messages: 87% ham, 13% spam)
- **Splits**: leakage-safe train/val/test splits created (grouped by exact text)
- **Baseline**: TF-IDF + Logistic Regression trained and evaluated (best performer)
- **Feature Engineering**: 19 heuristic features implemented and tested
- **Enhanced Model**: Random Forest + combined features (baseline still superior)
- **FastAPI Service**: Production-ready API with /score, /feedback, /health endpoints
- **AWS Deployment**: ECS Fargate with ECR, load balancer, and target groups

## API Usage

### Live Production API

The SMS Spam Detection API is deployed and accessible at:
**https://[your-deployment-url].ecs.us-east-2.on.aws**

### Endpoints

#### Health Check
```bash
curl https://[your-deployment-url].ecs.us-east-2.on.aws/health
```

#### Spam Detection
```bash
curl -X POST https://[your-deployment-url].ecs.us-east-2.on.aws/score \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT! Click here to claim your FREE prize NOW!", "message_id": "test-1"}'
```

Response:
```json
{
  "risk_score": 0.62,
  "is_threat": true,
  "model_version": "enhanced_model.joblib",
  "message_id": "test-1"
}
```

#### Feedback Submission
```bash
curl -X POST https://[your-deployment-url].ecs.us-east-2.on.aws/feedback \
  -H "Content-Type: application/json" \
  -d '{"message_id": "test-1", "text": "Sample message", "true_label": "spam"}'
```

## Local Development Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run API Locally
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run with Docker
```bash
# Build image
docker build -t sms-spam-api .

# Run container
docker run -d -p 8000:8000 --name sms-spam-test sms-spam-api

# Test locally
curl http://localhost:8000/health
```

## Testing

### Run Test Suite
```bash
pytest tests/ -v
```

### Manual API Testing
```bash
# Test spam message
curl -X POST https://[your-deployment-url].ecs.us-east-2.on.aws/score \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT! Click here to claim your FREE prize NOW!", "message_id": "test-spam"}'

# Expected: {"risk_score": 0.62, "is_threat": true, ...}

# Test legitimate message  
curl -X POST https://[your-deployment-url].ecs.us-east-2.on.aws/score \
  -H "Content-Type: application/json" \
  -d '{"text": "Hey, are we still meeting for lunch tomorrow?", "message_id": "test-ham"}'

# Expected: {"risk_score": 0.0, "is_threat": false, ...}
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

## Model Training

### Baseline Model

Train a TF-IDF + Logistic Regression baseline:

```bash
python training/train.py
```

Outputs:
- `artifacts/model.joblib`
- `reports/metrics.json`

### Enhanced Model

Train Random Forest with TF-IDF + heuristic features:

```bash
python training/train_enhanced.py
```

Outputs:
- `artifacts/enhanced_model.joblib`
- `reports/enhanced_metrics.json`

**S3 Storage**: `s3://sms-threat-demo-models-20260223/models/enhanced_model.joblib`

**Note**: Additional models from Day 1 comparison are also stored in S3:
- Logistic Regression: `s3://sms-threat-demo-models-20260223/models/logistic-regression/v1.0/model.joblib`
- Random Forest: `s3://sms-threat-demo-models-20260223/models/random-forest/v1.0/model.joblib`

## AWS Deployment Architecture

### Infrastructure Components

- **ECS Cluster**: `sms-spam-cluster` (Fargate)
- **ECS Service**: `sms-spam-detection-0636` 
- **Task Definition**: Revision 3 with correct port 8000 configuration
- **ECR Repository**: `[account-id].dkr.ecr.us-east-1.amazonaws.com/sms-spam-detection`
- **Load Balancer**: Application Load Balancer with health checks on port 8000
- **Public URL**: https://[your-deployment-url].ecs.us-east-2.on.aws

### Deployment Process

1. **Build and push Docker image**:
```bash
# Build for AMD64 platform (required for Fargate)
docker buildx build --platform linux/amd64 -t sms-spam-api .

# Tag for ECR
docker tag sms-spam-api:latest [account-id].dkr.ecr.us-east-1.amazonaws.com/sms-spam-detection:latest

# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [account-id].dkr.ecr.us-east-1.amazonaws.com

# Push to ECR
docker push [account-id].dkr.ecr.us-east-1.amazonaws.com/sms-spam-detection:latest
```

2. **Deploy via ECS Console**:
   - Create/update ECS service with Express Service feature
   - Configure container port 8000
   - Set health check path to `/health`
   - Update target group health checks to port 8000

### Key Configuration Notes

- **Container Port**: 8000 (FastAPI default)
- **Health Check Path**: `/health`
- **Platform**: linux/amd64 (required for Fargate)
- **Load Balancer**: Configured for port 8000 health checks

### Model Comparison

Compare baseline vs enhanced models:

```bash
python training/compare_models.py
```

Outputs:
- `artifacts/logistic_regression_model.joblib`
- `artifacts/random_forest_model.joblib`
- `reports/model_comparison.json`
- `reports/model_comparison.md`
