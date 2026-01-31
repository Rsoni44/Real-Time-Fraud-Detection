"""
Fraud Detection Cloud Function

Processes credit card transactions from Pub/Sub and makes real-time
fraud predictions using Vertex AI model.
"""

import json
import base64
import os
import functions_framework
from google.cloud import aiplatform
from google.cloud import bigquery
from datetime import datetime

# Config from environment
PROJECT_ID = os.environ.get('GCP_PROJECT', 'your-project-id')
LOCATION = os.environ.get('LOCATION', 'us-central1')
ENDPOINT_ID = os.environ.get('ENDPOINT_ID', 'your-endpoint-id')
TABLE_ID = os.environ.get('TABLE_ID', f'{PROJECT_ID}.fraud_dataset.predictions')
FRAUD_THRESHOLD = float(os.environ.get('FRAUD_THRESHOLD', '0.8'))

# Initialize clients once to avoid cold start overhead
aiplatform.init(project=PROJECT_ID, location=LOCATION)
bq_client = bigquery.Client(project=PROJECT_ID)

endpoint = None


def get_endpoint():
    """Cache the endpoint instance to reuse across function invocations"""
    global endpoint
    if endpoint is None:
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    return endpoint


@functions_framework.cloud_event
def process_transaction(cloud_event):
    """
    Main handler for incoming transaction messages.
    Decodes the transaction, runs prediction, logs to BigQuery.
    """
    try:
        # Decode the Pub/Sub message
        pubsub_message = base64.b64decode(cloud_event.data["message"]["data"])
        transaction = json.loads(pubsub_message)

        transaction_id = transaction.get('transaction_id', 'unknown')
        print(f"Processing transaction: {transaction_id}")

        # Extract features in the same order as training
        # This is critical - feature order must match exactly
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']

        missing_features = [f for f in feature_names if f not in transaction]
        if missing_features:
            error_msg = f"Missing features: {missing_features}"
            print(f"ERROR: {error_msg}")
            return {'error': error_msg, 'status': 'failed'}

        features = [float(transaction[f]) for f in feature_names]
        print(f"Extracted {len(features)} features, amount=${features[-1]:.2f}")

        # Call Vertex AI for prediction
        try:
            ep = get_endpoint()
            prediction_response = ep.predict(instances=[features])

            # Parse the response - format depends on model type
            if hasattr(prediction_response, 'predictions'):
                fraud_prediction = int(prediction_response.predictions[0])

                if hasattr(prediction_response, 'probabilities'):
                    fraud_probability = float(prediction_response.probabilities[0])
                elif len(prediction_response.predictions[0]) > 1:
                    fraud_probability = float(prediction_response.predictions[0][1])
                else:
                    fraud_probability = 1.0 if fraud_prediction == 1 else 0.0
            else:
                # AutoEncoder model returns reconstruction error
                reconstruction_error = float(prediction_response[0])
                fraud_prediction = 1 if reconstruction_error > FRAUD_THRESHOLD else 0
                fraud_probability = min(reconstruction_error / FRAUD_THRESHOLD, 1.0)

            print(f"Prediction: {fraud_prediction} (prob: {fraud_probability:.4f})")

        except Exception as e:
            error_msg = f"Vertex AI prediction failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'error': error_msg, 'status': 'failed'}

        # Log to BigQuery
        row_to_insert = {
            'transaction_id': transaction_id,
            'prediction': fraud_prediction,
            'fraud_probability': fraud_probability,
            'timestamp': datetime.utcnow().isoformat(),
        }

        for i, feature_name in enumerate(feature_names):
            column_name = feature_name.lower()
            row_to_insert[column_name] = features[i]

        errors = bq_client.insert_rows_json(TABLE_ID, [row_to_insert])

        if errors:
            print(f"WARNING: BigQuery insert errors: {errors}")
        else:
            print(f"Logged to BigQuery: {TABLE_ID}")

        # Alert on high fraud probability
        if fraud_probability >= FRAUD_THRESHOLD:
            alert_msg = (
                f"FRAUD ALERT - Transaction {transaction_id}\n"
                f"Probability: {fraud_probability:.2%}, Amount: ${features[-1]:.2f}\n"
                f"Time: {row_to_insert['timestamp']}"
            )
            print(alert_msg)
            # TODO: send to alerting system (Pub/Sub topic, webhook, etc)

        return {
            'status': 'success',
            'transaction_id': transaction_id,
            'fraud_prediction': fraud_prediction,
            'fraud_probability': fraud_probability,
            'logged_to_bigquery': errors is None or len(errors) == 0
        }

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {'error': error_msg, 'status': 'failed'}


def create_test_event(transaction_data):
    """Helper function for local testing - creates a mock cloud event"""
    import types

    message_data = json.dumps(transaction_data).encode('utf-8')
    encoded_data = base64.b64encode(message_data).decode('utf-8')

    mock_event = types.SimpleNamespace()
    mock_event.data = {
        "message": {
            "data": encoded_data
        }
    }

    return mock_event


if __name__ == "__main__":
    print("Testing Cloud Function locally...")

    # Create a sample transaction
    test_transaction = {
        "transaction_id": "test_local_001",
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }

    event = create_test_event(test_transaction)
    result = process_transaction(event)

    print("\nTest Result:")
    print(json.dumps(result, indent=2))
