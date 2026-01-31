# Real-Time Fraud Detection System on Google Cloud

## 1. Executive Summary
**Project Name:** Real-Time Fraud Detection Pipeline
**Role:** Machine Learning Engineer / Data Engineer
**Status:** Deployed (MVP)

### Problem Statement
Traditional fraud detection often relies on batch processing (e.g., analyzing transactions at the end of the day). In the modern financial landscape, fraud happens in milliseconds. A system is needed that can ingest, process, and flag fraudulent transactions in near real-time to prevent financial loss before it settles.

### Solution Overview
This project migrates a static Jupyter Notebook (batch analysis) into a fully decoupled, event-driven microservices architecture on Google Cloud Platform (GCP). It simulates a stream of credit card transactions, processes them via serverless functions, performs inference using a deployed Scikit-Learn model on Vertex AI, and persists data for post-incident analysis.

---

## 2. System Architecture

### High-Level Design
`[Transaction Source] -> [Pub/Sub] -> [Cloud Functions] -> [Vertex AI Endpoint] -> [BigQuery]`

### Component Breakdown

| Component | GCP Service | Role | Justification |
| :--- | :--- | :--- | :--- |
| **Event Source** | Local Python Script | Producer | Simulates a high-velocity stream of credit card terminal data using the `creditcard.csv` dataset. |
| **Ingestion Layer** | **Cloud Pub/Sub** | Message Queue | Decouples the producer from the consumer. Handles traffic spikes without crashing the inference engine. |
| **Processing Layer** | **Cloud Functions (Gen 2)** | Consumer / Orchestrator | Serverless compute that triggers on every message. It handles data transformation and invokes the model. |
| **Inference Layer** | **Vertex AI Prediction** | Model Serving | Provides a scalable, managed endpoint for the ML model (Random Forest/Autoencoder) with low latency. |
| **Data Sink** | **BigQuery** | Data Warehousing | Stores flagged transactions and raw logs for SQL-based analysis and model retraining pipelines. |

---

## 3. Technical Implementation

### 3.1 Model Training & Artifact Management
* **Source:** `Fraud_Detection_and_Prediction.ipynb`
* **Action:** The model (Random Forest) and Preprocessor (RobustScaler) are serialized using `joblib`.
* **Storage:** Artifacts (`model.pkl`, `scaler.pkl`) are uploaded to a Google Cloud Storage (GCS) bucket.

### 3.2 The Real-Time Pipeline
1.  **Ingestion:**
    * Transactions are published to the `transactions` Pub/Sub topic as JSON messages.
2.  **Trigger:**
    * The Pub/Sub subscription triggers a Cloud Function.
3.  **Transformation:**
    * The Cloud Function decodes the Base64 message.
    * It aligns features (`V1`...`V28`, `Amount`) to the exact order expected by the model.
4.  **Prediction:**
    * The function sends the vector to the **Vertex AI Endpoint**.
    * The endpoint returns a prediction (`0` or `1`) and a probability score.
5.  **Action:**
    * If `Fraud_Prob > 0.8`, the transaction is flagged.
    * All results are inserted into BigQuery table `fraud_dataset.predictions`.

---

## 4. Interview "Defense" Guide (System Design)

This section prepares you for specific questions during a System Design or ML Engineering interview.

### Q1: Why did you choose Pub/Sub over Kafka?
**Answer:** "For this specific implementation, I prioritized operational overhead and integration.
* **Operational Overhead:** Kafka requires managing brokers, Zookeeper, and partitions. Pub/Sub is fully managed and global by default.
* **Scale:** Pub/Sub scales automatically without me needing to re-partition topics.
* *Caveat:* If I needed strict ordering or log replayability for longer durations, Kafka would be better, but for immediate fraud alerting, Pub/Sub is sufficient."

### Q2: Why use Cloud Functions (Serverless) instead of a container on GKE/Kubernetes?
**Answer:** "The traffic pattern for fraud can be bursty.
* **Cost Efficiency:** Cloud Functions scale to zero when no transactions are happening, whereas GKE has a constant control plane cost.
* **Complexity:** For a single-purpose event handler (Pub/Sub trigger -> Inference), Kubernetes is over-engineering. Cloud Functions provide the simplest 'glue' code infrastructure."

### Q3: Why Vertex AI instead of just loading the model inside the Cloud Function?
**Answer:** "This is a critical distinction between 'Monolith' and 'Microservices'.
* **Decoupling:** If I load the model inside the function, every function invocation has to load the heavy model into memory (Cold Start), or I bloat the function size.
* **Scalability:** Vertex AI is optimized specifically for matrix operations and inference. It can scale independently of the ingestion layer.
* **Lifecycle:** I can update the model endpoint in Vertex AI without touching the Cloud Function code. It separates the Data Science lifecycle from the App Dev lifecycle."

### Q4: How would you handle Model Drift in this system?
**Answer:** "Currently, I log all predictions and ground truth (if available later) to BigQuery.
* **Monitoring:** I would set up a scheduled query in BigQuery or use Vertex AI Model Monitoring to compare the distribution of incoming features (e.g., `V1` mean) against the training baseline.
* **Retraining:** If drift is detected (e.g., KL Divergence exceeds a threshold), a Vertex AI Pipeline would trigger to retrain the model on the new data accumulated in BigQuery."

### Q5: How do you handle "Training-Serving Skew"?
**Answer:** "I explicitly serialized the `RobustScaler` object used during training and loaded that exact instance in the inference pipeline. This ensures that the raw 'Amount' value from the live transaction is scaled using the exact same median and quartile range as the training data, preventing skew."

---

## 5. Deployment Instructions (The "Kill Switch" Protocol)

**COST WARNING:** This project uses Vertex AI Endpoints which bill per hour even if idle.

### Prerequisites
1.  Google Cloud Project with Billing Enabled.
2.  `gcloud` CLI installed.
3.  Python 3.9+.

### Step 1: Deploy Infrastructure
```bash
# Enable APIs
gcloud services enable aiplatform.googleapis.com cloudfunctions.googleapis.com pubsub.googleapis.com bigquery.googleapis.com

# Create Pub/Sub Topic
gcloud pubsub topics create transactions

# Create BigQuery Dataset
bq mk fraud_dataset