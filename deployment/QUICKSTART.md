# Quick Start Guide: Deploy Fraud Detection in 30 Minutes

This guide will get you from the Jupyter notebook to a working production system on Google Cloud Platform.

## Prerequisites

- [ ] Google Cloud account with billing enabled
- [ ] `gcloud` CLI installed ([Install Guide](https://cloud.google.com/sdk/docs/install))
- [ ] Python 3.9+ installed
- [ ] creditcard.csv dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Step-by-Step Deployment

### Phase 1: Prepare Your Model (15 minutes)

#### 1.1 Extract Model Artifacts from Notebook

Open your `Fraud_Detection_and_Prediction.ipynb` notebook and add a new cell at the end:

```python
# Copy the entire contents of deployment/scripts/extract_model_artifacts.py
# and paste it into a new cell in your notebook

%run deployment/scripts/extract_model_artifacts.py
```

Or run it as a script:
```bash
# First, ensure you have the notebook's variables saved
jupyter nbconvert --to script Fraud-Detection/Fraud_Detection_and_Prediction.ipynb
python deployment/scripts/extract_model_artifacts.py
```

This creates an `artifacts/` directory with:
- `random_forest_model.pkl` - The trained model
- `robust_scaler.pkl` - Preprocessing scaler
- `metadata.json` - Model information

#### 1.2 Set Up GCP Project

```bash
# Set your project ID (replace with your actual project ID)
export PROJECT_ID="your-project-id"
export REGION="us-central1"

# Set active project
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  cloudfunctions.googleapis.com \
  pubsub.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com
```

#### 1.3 Upload Artifacts to Cloud Storage

```bash
# Create GCS bucket for model artifacts
gsutil mb -l $REGION gs://$PROJECT_ID-fraud-models/

# Upload artifacts
gsutil -m cp artifacts/* gs://$PROJECT_ID-fraud-models/

# Verify upload
gsutil ls gs://$PROJECT_ID-fraud-models/
```

---

### Phase 2: Set Up Infrastructure (5 minutes)

#### 2.1 Create Pub/Sub Topic

```bash
# Create topic for incoming transactions
gcloud pubsub topics create transactions

# Verify creation
gcloud pubsub topics list
```

#### 2.2 Create BigQuery Dataset and Table

```bash
# Create dataset
bq mk --dataset --location=US $PROJECT_ID:fraud_dataset

# Create predictions table
bq mk --table $PROJECT_ID:fraud_dataset.predictions \
  transaction_id:STRING,\
  prediction:INTEGER,\
  fraud_probability:FLOAT,\
  v1:FLOAT,v2:FLOAT,v3:FLOAT,v4:FLOAT,v5:FLOAT,v6:FLOAT,v7:FLOAT,v8:FLOAT,v9:FLOAT,v10:FLOAT,\
  v11:FLOAT,v12:FLOAT,v13:FLOAT,v14:FLOAT,v15:FLOAT,v16:FLOAT,v17:FLOAT,v18:FLOAT,v19:FLOAT,v20:FLOAT,\
  v21:FLOAT,v22:FLOAT,v23:FLOAT,v24:FLOAT,v25:FLOAT,v26:FLOAT,v27:FLOAT,v28:FLOAT,\
  amount:FLOAT,\
  timestamp:TIMESTAMP

# Verify table creation
bq show $PROJECT_ID:fraud_dataset.predictions
```

---

### Phase 3: Deploy Model to Vertex AI (10 minutes)

#### 3.1 Create Model Registry Entry

```bash
# For scikit-learn Random Forest model
gcloud ai models upload \
  --region=$REGION \
  --display-name=fraud-detection-rf \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest \
  --artifact-uri=gs://$PROJECT_ID-fraud-models/

# Save the model ID from the output
export MODEL_ID="<copy-model-id-from-output>"
```

#### 3.2 Create Endpoint

```bash
# Create endpoint
gcloud ai endpoints create \
  --region=$REGION \
  --display-name=fraud-detection-endpoint

# Save the endpoint ID from the output
export ENDPOINT_ID="<copy-endpoint-id-from-output>"
```

#### 3.3 Deploy Model to Endpoint

```bash
# This takes 10-15 minutes - go get coffee! ☕
gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --region=$REGION \
  --model=$MODEL_ID \
  --display-name=fraud-rf-v1 \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=3 \
  --traffic-split=0=100

# Monitor deployment status
gcloud ai endpoints describe $ENDPOINT_ID --region=$REGION
```

---

### Phase 4: Deploy Cloud Function (5 minutes)

#### 4.1 Navigate to Cloud Function Directory

```bash
cd deployment/cloud_function
```

#### 4.2 Deploy the Function

```bash
gcloud functions deploy process-fraud-transaction \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=process_transaction \
  --trigger-topic=transactions \
  --timeout=60s \
  --memory=512MB \
  --set-env-vars=ENDPOINT_ID=$ENDPOINT_ID,TABLE_ID=$PROJECT_ID.fraud_dataset.predictions,GCP_PROJECT=$PROJECT_ID

# Verify deployment
gcloud functions describe process-fraud-transaction \
  --region=$REGION \
  --gen2
```

---

### Phase 5: Test the System (5 minutes)

#### 5.1 Install Python Dependencies

```bash
cd ../scripts

# Install required packages
pip install google-cloud-pubsub pandas
```

#### 5.2 Publish Test Transactions

```bash
# Send 100 transactions at 10/second
python transaction_publisher.py \
  --project-id=$PROJECT_ID \
  --topic-id=transactions \
  --dataset=../../Fraud-Detection/creditcard.csv \
  --rate=10 \
  --limit=100 \
  --verbose

# Or test with fraud-only transactions
python transaction_publisher.py \
  --project-id=$PROJECT_ID \
  --fraud-only \
  --limit=50 \
  --verbose
```

#### 5.3 Monitor Cloud Function Logs

```bash
# In a separate terminal, watch the logs
gcloud functions logs read process-fraud-transaction \
  --region=$REGION \
  --gen2 \
  --limit=50 \
  --format="table(TIME_UTC, LOG)"

# Or tail logs in real-time
gcloud functions logs tail process-fraud-transaction \
  --region=$REGION \
  --gen2
```

#### 5.4 Query BigQuery Results

```bash
# Query recent predictions
bq query --use_legacy_sql=false \
"SELECT
  transaction_id,
  prediction,
  ROUND(fraud_probability, 4) as fraud_prob,
  amount,
  timestamp
FROM \`$PROJECT_ID.fraud_dataset.predictions\`
ORDER BY timestamp DESC
LIMIT 20;"

# Get fraud detection summary
bq query --use_legacy_sql=false \
"SELECT
  COUNT(*) as total_transactions,
  SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as flagged_as_fraud,
  ROUND(AVG(fraud_probability), 4) as avg_fraud_prob,
  ROUND(MAX(amount), 2) as max_amount
FROM \`$PROJECT_ID.fraud_dataset.predictions\`;"
```

---

## Verification Checklist

- [ ] Model artifacts uploaded to GCS
- [ ] Pub/Sub topic created
- [ ] BigQuery table created
- [ ] Vertex AI endpoint deployed and online
- [ ] Cloud Function deployed successfully
- [ ] Test transactions published
- [ ] Predictions appearing in BigQuery
- [ ] Fraud alerts in Cloud Function logs

---

## Cost Monitoring

### Expected Costs (24-hour test)
- **Vertex AI Endpoint:** ~$3-5/day (n1-standard-2)
- **Cloud Functions:** ~$0.10 (for 10,000 invocations)
- **BigQuery:** ~$0.01 (storage only)
- **Pub/Sub:** Free tier covers testing

### Cost-Saving Tips
```bash
# When not testing, undeploy the model to stop billing
gcloud ai endpoints undeploy-model $ENDPOINT_ID \
  --region=$REGION \
  --deployed-model-id=<deployed-model-id>

# Re-deploy when needed (takes 10-15 minutes)
```

---

## Troubleshooting

### Issue: "Model prediction failed"
**Solution:** Check that the scaler is correctly saved and uploaded to GCS.

```bash
# Verify scaler file exists
gsutil ls gs://$PROJECT_ID-fraud-models/robust_scaler.pkl

# Check Cloud Function logs for detailed error
gcloud functions logs read process-fraud-transaction --region=$REGION --limit=100
```

### Issue: "Missing features in transaction"
**Solution:** Ensure transaction_publisher.py includes all V1-V28 and Amount.

```bash
# Test with a single transaction
echo '{"transaction_id":"test","V1":-1.5,"V2":0.5,...,"Amount":100.0}' | \
  gcloud pubsub topics publish transactions --message=-
```

### Issue: "Endpoint not found"
**Solution:** Verify endpoint ID and ensure it's fully deployed.

```bash
# List all endpoints
gcloud ai endpoints list --region=$REGION

# Check deployment status
gcloud ai endpoints describe $ENDPOINT_ID --region=$REGION
```

### Issue: BigQuery insert errors
**Solution:** Check table schema matches the data.

```bash
# View table schema
bq show $PROJECT_ID:fraud_dataset.predictions

# Test insert manually
echo '[{"transaction_id":"test","prediction":0,"fraud_probability":0.1,"amount":100,"timestamp":"2024-01-01T00:00:00"}]' | \
  bq insert $PROJECT_ID:fraud_dataset.predictions
```

---

## Next Steps

1. **Monitor Model Performance:**
   - Set up BigQuery scheduled queries to track accuracy
   - Compare predictions to ground truth (ActualClass)
   - Calculate daily precision/recall metrics

2. **Implement Alerting:**
   - Send fraud alerts to a separate Pub/Sub topic
   - Trigger webhooks for high-value frauds
   - Integrate with incident management systems

3. **Model Retraining:**
   - Export predictions from BigQuery
   - Retrain model monthly with new data
   - Deploy new version to Vertex AI

4. **Scale Testing:**
   - Increase transaction rate to 100+/sec
   - Monitor Cloud Function cold starts
   - Optimize endpoint machine type

---

## Cleanup (When Done Testing)

**⚠️ IMPORTANT:** Run this to stop billing!

```bash
# 1. Undeploy model from endpoint (stops most billing)
gcloud ai endpoints undeploy-model $ENDPOINT_ID \
  --region=$REGION \
  --deployed-model-id=<deployed-model-id>

# 2. Delete endpoint
gcloud ai endpoints delete $ENDPOINT_ID --region=$REGION

# 3. Delete Cloud Function
gcloud functions delete process-fraud-transaction --region=$REGION --gen2

# 4. Delete Pub/Sub topic
gcloud pubsub topics delete transactions

# 5. Keep BigQuery data (costs ~$0.02/GB/month)
# Or delete if not needed:
# bq rm -r -f $PROJECT_ID:fraud_dataset
```

---

## Resources

- [Full Deployment Roadmap](../DEPLOYMENT_ROADMAP.md)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Functions Quickstart](https://cloud.google.com/functions/docs/quickstart)
- [BigQuery SQL Reference](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax)

---

**🎉 Congratulations!** You've deployed a real-time fraud detection system on GCP!

For questions or issues, check the troubleshooting section or refer to the full deployment roadmap.
