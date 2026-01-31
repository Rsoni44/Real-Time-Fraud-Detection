#!/bin/bash

#===============================================================================
# Automated Deployment Script for Fraud Detection System
# This script automates the deployment process outlined in QUICKSTART.md
#
# Usage:
#   ./deploy.sh --project-id your-project-id [--region us-central1]
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Model artifacts in artifacts/ directory
#   - Billing enabled on GCP project
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
REGION="us-central1"
PROJECT_ID=""
SKIP_VERTEX_AI=false

#===============================================================================
# Helper Functions
#===============================================================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}OK: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING:  $1${NC}"
}

print_info() {
    echo -e "${BLUE}$1${NC}"
}

#===============================================================================
# Parse Command Line Arguments
#===============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --skip-vertex-ai)
            SKIP_VERTEX_AI=true
            shift
            ;;
        --help)
            echo "Usage: $0 --project-id YOUR_PROJECT_ID [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --project-id ID     GCP Project ID (required)"
            echo "  --region REGION     GCP Region (default: us-central1)"
            echo "  --skip-vertex-ai    Skip Vertex AI deployment (for testing)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$PROJECT_ID" ]; then
    print_error "Project ID is required. Use --project-id YOUR_PROJECT_ID"
    exit 1
fi

#===============================================================================
# Pre-flight Checks
#===============================================================================

print_header "Pre-flight Checks"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI is not installed"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
print_success "gcloud CLI found"

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_error "Not authenticated with gcloud"
    echo "Run: gcloud auth login"
    exit 1
fi
print_success "gcloud authenticated"

# Set project
gcloud config set project $PROJECT_ID
print_success "Project set to: $PROJECT_ID"

#===============================================================================
# Phase 1: Enable APIs
#===============================================================================

print_header "Phase 1: Enabling Required APIs"

APIS=(
    "aiplatform.googleapis.com"
    "cloudfunctions.googleapis.com"
    "pubsub.googleapis.com"
    "bigquery.googleapis.com"
    "storage.googleapis.com"
    "cloudbuild.googleapis.com"
)

for api in "${APIS[@]}"; do
    print_info "Enabling $api..."
    gcloud services enable $api --project=$PROJECT_ID
done

print_success "All APIs enabled"

#===============================================================================
# Phase 2: Upload Model Artifacts to GCS
#===============================================================================

print_header "Phase 2: Uploading Model Artifacts"

BUCKET_NAME="${PROJECT_ID}-fraud-models"

# Check if artifacts directory exists
if [ ! -d "artifacts" ]; then
    print_warning "artifacts/ directory not found"
    print_info "Run the extract_model_artifacts.py script first"
    exit 1
fi

# Create GCS bucket
print_info "Creating GCS bucket: gs://$BUCKET_NAME/"
if gsutil mb -l $REGION gs://$BUCKET_NAME/ 2>/dev/null; then
    print_success "Bucket created"
else
    print_warning "Bucket already exists or creation failed"
fi

# Upload artifacts
print_info "Uploading model artifacts..."
gsutil -m cp artifacts/* gs://$BUCKET_NAME/
print_success "Artifacts uploaded to gs://$BUCKET_NAME/"

#===============================================================================
# Phase 3: Create Pub/Sub Topic
#===============================================================================

print_header "Phase 3: Creating Pub/Sub Topic"

TOPIC_NAME="transactions"

if gcloud pubsub topics create $TOPIC_NAME 2>/dev/null; then
    print_success "Pub/Sub topic created: $TOPIC_NAME"
else
    print_warning "Topic '$TOPIC_NAME' already exists"
fi

#===============================================================================
# Phase 4: Create BigQuery Dataset and Table
#===============================================================================

print_header "Phase 4: Creating BigQuery Infrastructure"

DATASET_NAME="fraud_dataset"
TABLE_NAME="predictions"

# Create dataset
print_info "Creating BigQuery dataset: $DATASET_NAME"
if bq mk --dataset --location=US $PROJECT_ID:$DATASET_NAME 2>/dev/null; then
    print_success "Dataset created"
else
    print_warning "Dataset '$DATASET_NAME' already exists"
fi

# Create table
print_info "Creating BigQuery table: $TABLE_NAME"
bq mk --table $PROJECT_ID:$DATASET_NAME.$TABLE_NAME \
  transaction_id:STRING,\
prediction:INTEGER,\
fraud_probability:FLOAT,\
v1:FLOAT,v2:FLOAT,v3:FLOAT,v4:FLOAT,v5:FLOAT,v6:FLOAT,v7:FLOAT,v8:FLOAT,v9:FLOAT,v10:FLOAT,\
v11:FLOAT,v12:FLOAT,v13:FLOAT,v14:FLOAT,v15:FLOAT,v16:FLOAT,v17:FLOAT,v18:FLOAT,v19:FLOAT,v20:FLOAT,\
v21:FLOAT,v22:FLOAT,v23:FLOAT,v24:FLOAT,v25:FLOAT,v26:FLOAT,v27:FLOAT,v28:FLOAT,\
amount:FLOAT,\
timestamp:TIMESTAMP 2>/dev/null || print_warning "Table already exists"

print_success "BigQuery infrastructure ready"

#===============================================================================
# Phase 5: Deploy to Vertex AI (Optional - takes 15+ minutes)
#===============================================================================

if [ "$SKIP_VERTEX_AI" = false ]; then
    print_header "Phase 5: Deploying Model to Vertex AI"
    print_warning "This step takes 15-20 minutes. Go get coffee! ☕"

    # Upload model
    print_info "Uploading model to Vertex AI..."
    MODEL_UPLOAD_OUTPUT=$(gcloud ai models upload \
        --region=$REGION \
        --display-name=fraud-detection-rf \
        --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest \
        --artifact-uri=gs://$BUCKET_NAME/ \
        --format="value(name)")

    MODEL_ID=$(echo $MODEL_UPLOAD_OUTPUT | awk -F'/' '{print $NF}')
    print_success "Model uploaded with ID: $MODEL_ID"

    # Create endpoint
    print_info "Creating Vertex AI endpoint..."
    ENDPOINT_OUTPUT=$(gcloud ai endpoints create \
        --region=$REGION \
        --display-name=fraud-detection-endpoint \
        --format="value(name)")

    ENDPOINT_ID=$(echo $ENDPOINT_OUTPUT | awk -F'/' '{print $NF}')
    print_success "Endpoint created with ID: $ENDPOINT_ID"

    # Deploy model to endpoint
    print_info "Deploying model to endpoint (this takes 10-15 minutes)..."
    gcloud ai endpoints deploy-model $ENDPOINT_ID \
        --region=$REGION \
        --model=$MODEL_ID \
        --display-name=fraud-rf-v1 \
        --machine-type=n1-standard-2 \
        --min-replica-count=1 \
        --max-replica-count=3 \
        --traffic-split=0=100

    print_success "Model deployed to endpoint"

    # Save endpoint ID for Cloud Function
    echo $ENDPOINT_ID > .endpoint_id
else
    print_warning "Skipping Vertex AI deployment (--skip-vertex-ai flag set)"
    ENDPOINT_ID="<manually-set-endpoint-id>"
fi

#===============================================================================
# Phase 6: Deploy Cloud Function
#===============================================================================

print_header "Phase 6: Deploying Cloud Function"

# Check if we have an endpoint ID
if [ -f ".endpoint_id" ]; then
    ENDPOINT_ID=$(cat .endpoint_id)
fi

if [ "$ENDPOINT_ID" = "<manually-set-endpoint-id>" ]; then
    print_warning "No Vertex AI endpoint ID available"
    print_info "Set ENDPOINT_ID environment variable before deploying Cloud Function"
    read -p "Enter Vertex AI Endpoint ID (or press Enter to skip): " USER_ENDPOINT_ID
    if [ ! -z "$USER_ENDPOINT_ID" ]; then
        ENDPOINT_ID=$USER_ENDPOINT_ID
    else
        print_warning "Skipping Cloud Function deployment"
        exit 0
    fi
fi

cd ../cloud_function

print_info "Deploying Cloud Function..."
gcloud functions deploy process-fraud-transaction \
    --gen2 \
    --runtime=python311 \
    --region=$REGION \
    --source=. \
    --entry-point=process_transaction \
    --trigger-topic=$TOPIC_NAME \
    --timeout=60s \
    --memory=512MB \
    --set-env-vars=ENDPOINT_ID=$ENDPOINT_ID,TABLE_ID=$PROJECT_ID.$DATASET_NAME.$TABLE_NAME,GCP_PROJECT=$PROJECT_ID

print_success "Cloud Function deployed"

cd ../scripts

#===============================================================================
# Deployment Summary
#===============================================================================

print_header "Deployment Summary"

echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo ""
echo "Resources Created:"
echo "  • GCS Bucket: gs://$BUCKET_NAME/"
echo "  • Pub/Sub Topic: $TOPIC_NAME"
echo "  • BigQuery Dataset: $DATASET_NAME"
echo "  • BigQuery Table: $TABLE_NAME"
if [ "$SKIP_VERTEX_AI" = false ]; then
    echo "  • Vertex AI Model ID: $MODEL_ID"
    echo "  • Vertex AI Endpoint ID: $ENDPOINT_ID"
fi
echo "  • Cloud Function: process-fraud-transaction"
echo ""

print_success "Deployment Complete!"

#===============================================================================
# Next Steps
#===============================================================================

print_header "Next Steps"

echo "1. Test the system:"
echo "   python transaction_publisher.py --project-id=$PROJECT_ID --limit=100"
echo ""
echo "2. Monitor Cloud Function logs:"
echo "   gcloud functions logs tail process-fraud-transaction --region=$REGION --gen2"
echo ""
echo "3. Query BigQuery results:"
echo "   bq query --use_legacy_sql=false 'SELECT * FROM \`$PROJECT_ID.$DATASET_NAME.$TABLE_NAME\` ORDER BY timestamp DESC LIMIT 10'"
echo ""

print_warning "IMPORTANT: Vertex AI endpoints bill per hour (~$3-5/day)"
print_info "To stop billing, undeploy the model:"
echo "   gcloud ai endpoints undeploy-model $ENDPOINT_ID --region=$REGION --deployed-model-id=<id>"
echo ""

print_success "Happy fraud detecting! "
