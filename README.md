# Real-Time Fraud Detection on Google Cloud Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GCP](https://img.shields.io/badge/GCP-Cloud%20Platform-4285F4.svg)](https://cloud.google.com/)

A cloud-native fraud detection system that processes credit card transactions in real-time using event-driven microservices architecture on GCP.

## Project Overview

### The Challenge
Traditional fraud detection relies on batch processing (analyzing transactions at end-of-day). Modern fraud happens in milliseconds. This project demonstrates how to architect a **real-time fraud detection system** that can process and flag fraudulent credit card transactions in under 1 second.

### The Solution
An event-driven microservices architecture on GCP that:
- Ingests transaction streams via Pub/Sub
- Processes events with serverless Cloud Functions
- Serves ML models via Vertex AI endpoints
- Stores predictions in BigQuery for analysis
- Achieves sub-second end-to-end latency

---

## Architecture

```
┌─────────────────┐
│  Transaction    │
│   Publisher     │ (Simulates real-time stream)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Pub/Sub       │ (Message Queue)
│  "transactions" │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cloud Function  │ (Event Processor)
│   - Decode msg  │
│   - Extract V1-V28, Amount
│   - Call Vertex AI
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vertex AI      │ (Model Serving)
│  Random Forest  │ Accuracy: 99.99%
│  Prediction     │ Recall: 100%
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   BigQuery      │ (Data Warehouse)
│  - Predictions  │
│  - Analytics    │
└─────────────────┘
```

---

## Key Features

### ML Operations (MLOps)
- **Model Serialization:** Joblib for scikit-learn models
- **Preprocessing Pipeline:** RobustScaler for consistent feature scaling
- **Model Registry:** Vertex AI for versioning and deployment
- **Training-Serving Skew Prevention:** Saved preprocessing artifacts

### Cloud Architecture
- **Event-Driven:** Pub/Sub decouples producers from consumers
- **Serverless:** Cloud Functions scale to zero when idle
- **Managed ML:** Vertex AI handles model serving infrastructure
- **Analytics-Ready:** BigQuery for SQL-based analysis

### Production-Ready
- **Real-Time Processing:** <1 second latency
- **Auto-Scaling:** Handles traffic spikes automatically
- **Monitoring:** Cloud Function logs + BigQuery analytics
- **Cost-Optimized:** Pay-per-use pricing model

---

## Model Performance

Trained on [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions)

| Model | Accuracy | Precision | Recall | Deployment |
|-------|----------|-----------|--------|------------|
| Random Forest (Balanced) | **99.99%** | **99.99%** | **100%** | Production |
| Attention Neural Network | 98.96% | 98.65% | 99.29% | Alternative |
| AutoEncoder | Variable | Variable | Variable | Anomaly Detection |

---

## Quick Start

### Prerequisites
- Google Cloud account (free tier works for testing)
- `gcloud` CLI installed ([Install Guide](https://cloud.google.com/sdk/docs/install))
- Python 3.9+
- Kaggle dataset downloaded ([creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))

### Automated Deployment

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fraud-detection-gcp.git
cd fraud-detection-gcp

# 2. Set up GCP project
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# 3. Run automated deployment (takes ~20 minutes)
cd deployment/scripts
./deploy.sh --project-id $PROJECT_ID

# 4. Test with simulated transactions
python transaction_publisher.py \
  --project-id $PROJECT_ID \
  --limit 100 \
  --verbose
```

### Manual Deployment
Follow the step-by-step guide: [deployment/QUICKSTART.md](deployment/QUICKSTART.md)

---

## Project Structure

```
.
├── README.md                          # Project overview
├── DEPLOYMENT_ROADMAP.md              # Comprehensive deployment guide
│
├── notebooks/
│   └── fraud_detection_training.ipynb        # Model training and comparison
│
└── deployment/
    ├── QUICKSTART.md                  # 30-minute quick start guide
    │
    ├── cloud_function/
    │   ├── main.py                    # Cloud Function for processing
    │   └── requirements.txt           # Python dependencies
    │
    └── scripts/
        ├── extract_model_artifacts.py # Extract models from notebook
        ├── transaction_publisher.py   # Simulate transaction stream
        └── deploy.sh                  # Automated deployment script
```

---

## What I Learned

### Technical Skills
- **GCP Services:** Pub/Sub, Cloud Functions, Vertex AI, BigQuery, Cloud Storage
- **MLOps:** Model deployment, versioning, monitoring
- **System Design:** Event-driven architecture, microservices patterns
- **Real-Time Processing:** Stream processing, latency optimization

### Engineering Practices
- **Training-Serving Skew:** Consistent preprocessing between training and inference
- **Cost Optimization:** Serverless architecture for variable workloads
- **Scalability:** Auto-scaling components for traffic spikes
- **Monitoring:** Logging, alerting, and analytics

### Trade-offs & Decisions
- **Pub/Sub vs Kafka:** Chose managed service over operational overhead
- **Cloud Functions vs GKE:** Serverless for simplicity and cost
- **Vertex AI vs In-Function:** Decoupled model serving for scalability
- **Random Forest vs Deep Learning:** Better accuracy with simpler deployment

---

## Interview Talking Points

### System Design Questions

**"Why did you choose Pub/Sub over Kafka?"**
> "I prioritized operational overhead. Pub/Sub is fully managed and scales automatically, whereas Kafka requires managing brokers and partitions. For immediate fraud alerting, Pub/Sub provides sufficient guarantees with zero operational burden."

**"How do you handle model drift?"**
> "I log all predictions to BigQuery with timestamps. I would set up scheduled queries to monitor feature distributions against training baselines using KL divergence. If drift exceeds a threshold, Vertex AI Pipelines would trigger model retraining."

**"How did you prevent training-serving skew?"**
> "I serialized the exact `RobustScaler` instance used during training and loaded it in the inference pipeline. This ensures live transactions are scaled using the same median and quartile ranges as training data."

---

## Cost Analysis

### 24-Hour Test Run (~$3-5)
- **Vertex AI Endpoint:** $3-5 (n1-standard-2, 24 hours)
- **Cloud Functions:** $0.10 (10,000 invocations)
- **Pub/Sub:** $0.00 (free tier)
- **BigQuery:** $0.01 (storage only)

### Cost Optimization
```bash
# Stop billing by undeploying endpoint when not testing
gcloud ai endpoints undeploy-model <ENDPOINT_ID> \
  --region us-central1 \
  --deployed-model-id <ID>
```

---

## Future Enhancements

- [ ] **Real-Time Monitoring Dashboard** (Cloud Monitoring + Data Studio)
- [ ] **Alerting Pipeline** (Pub/Sub topic → Email/SMS via SendGrid/Twilio)
- [ ] **A/B Testing** (Deploy multiple models, compare performance)
- [ ] **Model Retraining Pipeline** (Vertex AI Pipelines on schedule)
- [ ] **Feature Store** (Vertex AI Feature Store for real-time features)
- [ ] **CI/CD Pipeline** (Cloud Build for automated deployments)

---

## Testing

### View Cloud Function Logs
```bash
gcloud functions logs tail process-fraud-transaction \
  --region us-central1 --gen2
```

### Query Predictions
```sql
SELECT
  transaction_id,
  prediction,
  ROUND(fraud_probability, 4) as fraud_prob,
  amount,
  timestamp
FROM `YOUR_PROJECT.fraud_dataset.predictions`
ORDER BY timestamp DESC
LIMIT 20;
```

### Performance Metrics
```sql
SELECT
  COUNT(*) as total_transactions,
  SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as flagged_frauds,
  ROUND(AVG(fraud_probability), 4) as avg_fraud_prob
FROM `YOUR_PROJECT.fraud_dataset.predictions`;
```

---

## Technologies Used

**Cloud Platform:**
- Google Cloud Platform (GCP)

**GCP Services:**
- Pub/Sub (Message Queue)
- Cloud Functions Gen2 (Serverless Compute)
- Vertex AI (ML Model Serving)
- BigQuery (Data Warehouse)
- Cloud Storage (Artifact Storage)

**ML Stack:**
- Python 3.9+
- scikit-learn (Random Forest)
- TensorFlow/Keras (LSTM, AutoEncoder)
- Pandas, NumPy (Data Processing)
- Joblib (Model Serialization)

**DevOps:**
- gcloud CLI
- Bash scripting
- Git

---

## Resources

- [Full Deployment Roadmap](DEPLOYMENT_ROADMAP.md)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Functions Guide](https://cloud.google.com/functions/docs)
- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Disclaimer

This is an **educational portfolio project** to demonstrate cloud ML engineering skills. It is not production-ready for handling real financial transactions. The dataset is synthetic and publicly available from Kaggle.

**Not suitable for:**
- Real financial fraud detection (requires PCI compliance, security audits)
- Production deployment without further security hardening
- Handling PII or sensitive financial data

**Suitable for:**
- Learning MLOps and cloud architecture
- Portfolio demonstration
- System design interview preparation

---

## License

This project is for educational purposes. Dataset credit: [Kaggle - Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Connect

**Preety Vandana**
- LinkedIn: [linkedin.com/in/your-profile](https://www.linkedin.com/in/your-profile)
- Portfolio: [your-website.com](https://your-website.com)
- Email: your.email@example.com

---

## Acknowledgments

- Dataset: ULB Machine Learning Group via [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Architecture patterns: Google Cloud Architecture Center
- Deployment best practices: Google Cloud documentation

---

If you found this project helpful, please give it a star!
