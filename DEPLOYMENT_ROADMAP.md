# Fraud Detection: From Notebook to Production Deployment

## Current State vs Target State Analysis

### What You Have (Current State)
**Location:** `Fraud-Detection/Fraud_Detection_and_Prediction.ipynb`

**Type:** Jupyter Notebook with batch ML training and evaluation

**Components:**
- 9 trained ML models (Logistic Regression, Random Forest, SVM, LSTM, Attention NN, AutoEncoder)
- Data preprocessing pipeline (RobustScaler, SMOTE balancing)
- Model evaluation and comparison
- Static analysis on creditcard.csv dataset

**Best Performing Models:**
1. **Random Forest (Balanced):** 99.99% accuracy, 100% recall
2. **Attention Neural Network:** 98.96% accuracy, 99.29% recall
3. **AutoEncoder:** Most production-ready (anomaly detection, flexible thresholding)

### What Your PRD Describes (Target State)
**Type:** Real-time event-driven microservices architecture on GCP

**Components:**
- Transaction stream ingestion (Pub/Sub)
- Serverless processing (Cloud Functions)
- Model serving (Vertex AI)
- Data warehousing (BigQuery)
- Near real-time fraud detection (<1 second latency)

### The Gap: Batch → Real-Time
Your notebook trains models in batch mode. Your PRD envisions a streaming production system. This roadmap bridges that gap.

---

## Phase 1: Model Selection & Artifact Preparation

### Step 1.1: Choose Your Production Model
Based on the notebook analysis, I recommend:

**Option A: Random Forest (Recommended for MVP)**
- Pros: Best accuracy (99.99%), fast inference, no GPU needed
- Cons: Requires balanced training data (SMOTE)
- Use Case: When you need highest precision

**Option B: AutoEncoder (Recommended for Production)**
- Pros: Anomaly detection, flexible threshold, unsupervised
- Cons: Slightly lower accuracy, requires MinMaxScaler
- Use Case: When fraud patterns evolve over time

### Step 1.2: Extract and Save Model Artifacts
You need to create a new notebook cell to serialize these objects:

```python
import joblib
import pickle
from keras.models import load_model

# Option A: Save Random Forest artifacts
# 1. Save the trained Random Forest model
joblib.dump(rf_model, 'model.pkl')

# 2. Save the RobustScaler (critical for preprocessing!)
joblib.dump(rob_scaler, 'scaler.pkl')

# Option B: Save AutoEncoder artifacts
# 1. Save the AutoEncoder model
autoencoder.save('autoencoder.h5')

# 2. Save the MinMaxScaler
joblib.dump(scaler, 'minmax_scaler.pkl')

# 3. Save the anomaly threshold
with open('threshold.pkl', 'wb') as f:
    pickle.dump(1.57, f)  # 99th percentile threshold
```

### Step 1.3: Upload Artifacts to Google Cloud Storage
```bash
# Create GCS bucket for model artifacts
gsutil mb gs://fraud-detection-models-[YOUR-PROJECT-ID]/

# Upload artifacts
gsutil cp model.pkl gs://fraud-detection-models-[YOUR-PROJECT-ID]/
gsutil cp scaler.pkl gs://fraud-detection-models-[YOUR-PROJECT-ID]/
gsutil cp autoencoder.h5 gs://fraud-detection-models-[YOUR-PROJECT-ID]/
gsutil cp minmax_scaler.pkl gs://fraud-detection-models-[YOUR-PROJECT-ID]/
```

---

## Phase 2: Infrastructure Setup on GCP

### Step 2.1: Enable Required APIs
```bash
gcloud services enable aiplatform.googleapis.com \
  cloudfunctions.googleapis.com \
  pubsub.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com
```

### Step 2.2: Create Pub/Sub Topic & Subscription
```bash
# Create topic for incoming transactions
gcloud pubsub topics create transactions

# Create subscription (Cloud Function will auto-create this, but explicit is better)
gcloud pubsub subscriptions create transactions-sub \
  --topic=transactions \
  --ack-deadline=60
```

### Step 2.3: Create BigQuery Dataset & Table
```bash
# Create dataset
bq mk --dataset --location=US fraud_dataset

# Create predictions table
bq mk --table fraud_dataset.predictions \
  transaction_id:STRING,\
  prediction:INTEGER,\
  fraud_probability:FLOAT,\
  v1:FLOAT,v2:FLOAT,v3:FLOAT,v4:FLOAT,v5:FLOAT,\
  v6:FLOAT,v7:FLOAT,v8:FLOAT,v9:FLOAT,v10:FLOAT,\
  v11:FLOAT,v12:FLOAT,v13:FLOAT,v14:FLOAT,v15:FLOAT,\
  v16:FLOAT,v17:FLOAT,v18:FLOAT,v19:FLOAT,v20:FLOAT,\
  v21:FLOAT,v22:FLOAT,v23:FLOAT,v24:FLOAT,v25:FLOAT,\
  v26:FLOAT,v27:FLOAT,v28:FLOAT,\
  amount:FLOAT,\
  timestamp:TIMESTAMP
```

---

## Phase 3: Model Deployment to Vertex AI

### Step 3.1: Create Custom Prediction Container (Optional but Recommended)
Since you have preprocessing (RobustScaler/MinMaxScaler), you need a custom predictor:

**Create `predictor.py`:**
```python
import os
import joblib
import numpy as np
from google.cloud import storage

class FraudPredictor:
    def __init__(self):
        """Load model and scaler from GCS during container startup."""
        # Download artifacts from GCS
        bucket_name = os.environ.get('MODEL_BUCKET', 'fraud-detection-models')
        self.model = self._load_from_gcs(bucket_name, 'model.pkl')
        self.scaler = self._load_from_gcs(bucket_name, 'scaler.pkl')

    def _load_from_gcs(self, bucket_name, blob_name):
        """Helper to download and load pickle files from GCS."""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(f'/tmp/{blob_name}')
        return joblib.load(f'/tmp/{blob_name}')

    def preprocess(self, instances):
        """Apply the same preprocessing as training."""
        # instances = [[V1, V2, ..., V28, Amount], ...]
        instances = np.array(instances)

        # Scale the Amount (last column)
        instances[:, -1] = self.scaler.transform(instances[:, -1].reshape(-1, 1)).flatten()

        return instances

    def predict(self, instances):
        """Run prediction on preprocessed data."""
        processed = self.preprocess(instances)
        predictions = self.model.predict(processed)
        probabilities = self.model.predict_proba(processed)[:, 1]  # Fraud probability

        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
```

### Step 3.2: Deploy to Vertex AI Endpoint
```bash
# Upload model to Vertex AI Model Registry
gcloud ai models upload \
  --region=us-central1 \
  --display-name=fraud-detection-rf \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest \
  --artifact-uri=gs://fraud-detection-models-[YOUR-PROJECT-ID]/

# Create endpoint
gcloud ai endpoints create \
  --region=us-central1 \
  --display-name=fraud-detection-endpoint

# Deploy model to endpoint (this takes 10-15 minutes)
gcloud ai endpoints deploy-model [ENDPOINT-ID] \
  --region=us-central1 \
  --model=[MODEL-ID] \
  --display-name=fraud-detection-v1 \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=3
```

---

## Phase 4: Cloud Function for Real-Time Processing

### Step 4.1: Create Cloud Function Code
**Create `main.py`:**
```python
import json
import base64
import functions_framework
from google.cloud import aiplatform
from google.cloud import bigquery
from datetime import datetime

# Initialize clients (done once per cold start)
aiplatform.init(project='[YOUR-PROJECT-ID]', location='us-central1')
bq_client = bigquery.Client()

ENDPOINT_ID = '[YOUR-ENDPOINT-ID]'
TABLE_ID = '[YOUR-PROJECT-ID].fraud_dataset.predictions'

@functions_framework.cloud_event
def process_transaction(cloud_event):
    """Triggered by Pub/Sub message containing transaction data."""

    # Decode Pub/Sub message
    pubsub_message = base64.b64decode(cloud_event.data["message"]["data"])
    transaction = json.loads(pubsub_message)

    # Extract features in exact order: V1-V28, Amount
    features = [
        transaction['V1'], transaction['V2'], transaction['V3'], transaction['V4'],
        transaction['V5'], transaction['V6'], transaction['V7'], transaction['V8'],
        transaction['V9'], transaction['V10'], transaction['V11'], transaction['V12'],
        transaction['V13'], transaction['V14'], transaction['V15'], transaction['V16'],
        transaction['V17'], transaction['V18'], transaction['V19'], transaction['V20'],
        transaction['V21'], transaction['V22'], transaction['V23'], transaction['V24'],
        transaction['V25'], transaction['V26'], transaction['V27'], transaction['V28'],
        transaction['Amount']
    ]

    # Call Vertex AI endpoint
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    prediction = endpoint.predict(instances=[features])

    # Parse prediction
    fraud_prediction = int(prediction.predictions[0])
    fraud_probability = float(prediction.probabilities[0])

    # Log to BigQuery
    row = {
        'transaction_id': transaction.get('transaction_id', 'unknown'),
        'prediction': fraud_prediction,
        'fraud_probability': fraud_probability,
        'v1': features[0], 'v2': features[1], 'v3': features[2], 'v4': features[3],
        'v5': features[4], 'v6': features[5], 'v7': features[6], 'v8': features[7],
        'v9': features[8], 'v10': features[9], 'v11': features[10], 'v12': features[11],
        'v13': features[12], 'v14': features[13], 'v15': features[14], 'v16': features[15],
        'v17': features[16], 'v18': features[17], 'v19': features[18], 'v20': features[19],
        'v21': features[20], 'v22': features[21], 'v23': features[22], 'v24': features[23],
        'v25': features[24], 'v26': features[25], 'v27': features[26], 'v28': features[27],
        'amount': features[28],
        'timestamp': datetime.utcnow().isoformat()
    }

    errors = bq_client.insert_rows_json(TABLE_ID, [row])

    if errors:
        print(f"BigQuery insert errors: {errors}")

    # Alert if high fraud probability
    if fraud_probability > 0.8:
        print(f"FRAUD ALERT: Transaction {row['transaction_id']} - {fraud_probability:.2%} probability")

    return {'status': 'success', 'fraud_probability': fraud_probability}
```

**Create `requirements.txt`:**
```
google-cloud-aiplatform==1.38.0
google-cloud-bigquery==3.14.0
functions-framework==3.*
```

### Step 4.2: Deploy Cloud Function
```bash
gcloud functions deploy process-fraud-transaction \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=process_transaction \
  --trigger-topic=transactions \
  --set-env-vars=ENDPOINT_ID=[YOUR-ENDPOINT-ID] \
  --timeout=60s \
  --memory=512MB
```

---

## Phase 5: Transaction Producer (Simulation)

### Step 5.1: Create Transaction Publisher
**Create `transaction_publisher.py`:**
```python
import json
import time
import pandas as pd
from google.cloud import pubsub_v1

# Load the creditcard.csv dataset
df = pd.read_csv('Fraud-Detection/creditcard.csv')

# Initialize Pub/Sub publisher
project_id = '[YOUR-PROJECT-ID]'
topic_id = 'transactions'
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

print(f"Publishing transactions to {topic_path}")

# Simulate real-time stream
for idx, row in df.iterrows():
    # Create transaction message
    transaction = {
        'transaction_id': f'txn_{idx}',
        'V1': float(row['V1']),
        'V2': float(row['V2']),
        'V3': float(row['V3']),
        'V4': float(row['V4']),
        'V5': float(row['V5']),
        'V6': float(row['V6']),
        'V7': float(row['V7']),
        'V8': float(row['V8']),
        'V9': float(row['V9']),
        'V10': float(row['V10']),
        'V11': float(row['V11']),
        'V12': float(row['V12']),
        'V13': float(row['V13']),
        'V14': float(row['V14']),
        'V15': float(row['V15']),
        'V16': float(row['V16']),
        'V17': float(row['V17']),
        'V18': float(row['V18']),
        'V19': float(row['V19']),
        'V20': float(row['V20']),
        'V21': float(row['V21']),
        'V22': float(row['V22']),
        'V23': float(row['V23']),
        'V24': float(row['V24']),
        'V25': float(row['V25']),
        'V26': float(row['V26']),
        'V27': float(row['V27']),
        'V28': float(row['V28']),
        'Amount': float(row['Amount']),
        'ActualClass': int(row['Class'])  # Ground truth for later evaluation
    }

    # Publish to Pub/Sub
    message_json = json.dumps(transaction)
    future = publisher.publish(topic_path, message_json.encode('utf-8'))

    print(f"Published transaction {idx}: {future.result()}")

    # Simulate realistic transaction rate (adjust as needed)
    time.sleep(0.1)  # 10 transactions/second

print("All transactions published successfully")
```

### Step 5.2: Run the Publisher
```bash
python transaction_publisher.py
```

---

## Phase 6: Monitoring & Validation

### Step 6.1: Monitor Cloud Function Logs
```bash
gcloud functions logs read process-fraud-transaction \
  --region=us-central1 \
  --limit=50
```

### Step 6.2: Query BigQuery for Results
```sql
-- View recent predictions
SELECT
  transaction_id,
  prediction,
  fraud_probability,
  amount,
  timestamp
FROM `fraud_dataset.predictions`
ORDER BY timestamp DESC
LIMIT 100;

-- Fraud alert summary
SELECT
  COUNT(*) as total_transactions,
  SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as flagged_frauds,
  AVG(fraud_probability) as avg_fraud_prob
FROM `fraud_dataset.predictions`
WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR);
```

### Step 6.3: Validate Model Performance
```sql
-- If you included ActualClass in your messages:
SELECT
  SUM(CASE WHEN prediction = actual_class THEN 1 ELSE 0 END) / COUNT(*) as accuracy,
  SUM(CASE WHEN prediction = 1 AND actual_class = 1 THEN 1 ELSE 0 END) /
    NULLIF(SUM(CASE WHEN actual_class = 1 THEN 1 ELSE 0 END), 0) as recall,
  SUM(CASE WHEN prediction = 1 AND actual_class = 1 THEN 1 ELSE 0 END) /
    NULLIF(SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END), 0) as precision
FROM `fraud_dataset.predictions`;
```

---

## Phase 7: Cost Optimization & Shutdown

### Important: Vertex AI Endpoints Bill by the Hour
Even when idle, a deployed endpoint costs ~$50-100/month per replica.

### Shutdown Checklist
```bash
# 1. Undeploy model from endpoint
gcloud ai endpoints undeploy-model [ENDPOINT-ID] \
  --region=us-central1 \
  --deployed-model-id=[DEPLOYED-MODEL-ID]

# 2. Delete endpoint
gcloud ai endpoints delete [ENDPOINT-ID] --region=us-central1

# 3. Delete Cloud Function
gcloud functions delete process-fraud-transaction \
  --region=us-central1

# 4. Delete Pub/Sub topic
gcloud pubsub topics delete transactions

# 5. Keep BigQuery data for analysis (minimal cost)
# BigQuery storage is ~$0.02/GB/month
```

---

## Key Differences: Notebook vs Production

| Aspect | Notebook (Current) | Production (Target) |
|--------|-------------------|---------------------|
| **Data Input** | Static CSV file | Real-time Pub/Sub stream |
| **Processing** | Batch (all at once) | Event-driven (per transaction) |
| **Latency** | Minutes/hours | <1 second |
| **Scalability** | Limited by local resources | Auto-scales with traffic |
| **Model Updates** | Retrain entire notebook | Deploy new version to Vertex AI |
| **Monitoring** | Matplotlib plots | BigQuery + Cloud Monitoring |
| **Preprocessing** | Done once in notebook | Must happen per transaction |

---

## Critical Success Factors

### 1. Training-Serving Skew Prevention
The #1 cause of production ML failures is when preprocessing differs between training and serving.

**Solution:** Save and load the EXACT SAME scaler object:
```python
# Training (Notebook)
rob_scaler.fit(training_data)
joblib.dump(rob_scaler, 'scaler.pkl')

# Serving (Cloud Function/Vertex AI)
rob_scaler = joblib.load('scaler.pkl')
scaled = rob_scaler.transform(live_data)
```

### 2. Feature Order Consistency
Your model expects features in a specific order: V1, V2, ..., V28, Amount.

**Validation:** Add assertions in your Cloud Function:
```python
assert len(features) == 29, "Expected 29 features"
assert 'V1' in transaction and 'V28' in transaction, "Missing V features"
```

### 3. Model Versioning
Use Vertex AI's model registry to track versions:
```bash
gcloud ai models upload \
  --display-name=fraud-detection-rf \
  --version-aliases=v1,latest
```

---

## Next Steps: Your Learning Path

1. **Start Simple:** Run the notebook locally, save the Random Forest model
2. **Test Locally:** Write a Python script that loads model.pkl and scaler.pkl, makes predictions
3. **Deploy to GCP:** Follow Phase 2-4 step-by-step
4. **Simulate Traffic:** Run the transaction publisher
5. **Monitor:** Check Cloud Function logs and BigQuery
6. **Iterate:** Try deploying the AutoEncoder, compare performance

---

## Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Functions Python Quickstart](https://cloud.google.com/functions/docs/quickstart-python)
- [Pub/Sub Best Practices](https://cloud.google.com/pubsub/docs/best-practices)
- [Scikit-learn Model Persistence](https://scikit-learn.org/stable/model_persistence.html)

---

**Estimated Timeline:**
- Phase 1-2: 1-2 hours (setup)
- Phase 3: 30 minutes + 15 minute deployment wait
- Phase 4: 30 minutes
- Phase 5-6: 1 hour (testing & validation)

**Total Cost (24-hour test):**
- Vertex AI Endpoint: ~$3-5/day
- Cloud Functions: ~$0.10 (for thousands of transactions)
- BigQuery: ~$0.01 (storage)
- Pub/Sub: Free tier covers testing

**Pro Tip:** Always undeploy your Vertex AI endpoint when not actively testing to avoid hourly charges!
