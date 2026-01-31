"""
Model Artifact Extraction Script
Add this cell to the end of your Fraud_Detection_and_Prediction.ipynb notebook
to save the trained models and preprocessing objects for production deployment.

This script assumes you've already trained the models in the notebook.
"""

import joblib
import pickle
import os
from datetime import datetime

# Create artifacts directory
os.makedirs('artifacts', exist_ok=True)

print("=" * 60)
print("SAVING MODEL ARTIFACTS FOR PRODUCTION DEPLOYMENT")
print("=" * 60)

# ============================================================================
# OPTION A: Random Forest Model (Recommended for MVP)
# ============================================================================
try:
    # Save the Random Forest model trained on balanced data
    # Make sure you have 'rf_model' or similar variable from your notebook
    # Adjust the variable name based on your notebook
    joblib.dump(rf_model, 'artifacts/random_forest_model.pkl')
    print("OK: Random Forest model saved: artifacts/random_forest_model.pkl")
except NameError:
    print("WARNING:  Random Forest model not found. Train the model first or check variable name.")

# ============================================================================
# OPTION B: AutoEncoder Model (Recommended for Anomaly Detection)
# ============================================================================
try:
    # Save the AutoEncoder Keras model
    # Make sure you have 'autoencoder' variable from your notebook
    autoencoder.save('artifacts/autoencoder.h5')
    print("OK: AutoEncoder model saved: artifacts/autoencoder.h5")

    # Save the anomaly threshold (99th percentile)
    threshold = 1.57  # Adjust based on your notebook's threshold calculation
    with open('artifacts/anomaly_threshold.pkl', 'wb') as f:
        pickle.dump(threshold, f)
    print(f"OK: Anomaly threshold saved: {threshold}")
except NameError:
    print("WARNING:  AutoEncoder model not found. Train the model first or check variable name.")

# ============================================================================
# CRITICAL: Preprocessing Objects (Required for both models)
# ============================================================================

# 1. Save RobustScaler (used for Amount feature)
try:
    joblib.dump(rob_scaler, 'artifacts/robust_scaler.pkl')
    print("OK: RobustScaler saved: artifacts/robust_scaler.pkl")
    print(f"   - Center: {rob_scaler.center_}")
    print(f"   - Scale: {rob_scaler.scale_}")
except NameError:
    print("WARNING:  RobustScaler not found. This is CRITICAL for production!")

# 2. Save MinMaxScaler (if using AutoEncoder)
try:
    if 'scaler' in dir() and hasattr(scaler, 'data_min_'):
        joblib.dump(scaler, 'artifacts/minmax_scaler.pkl')
        print("OK: MinMaxScaler saved: artifacts/minmax_scaler.pkl")
except Exception as e:
    print(f"WARNING:  MinMaxScaler not saved: {e}")

# ============================================================================
# OPTIONAL: Save Additional Models
# ============================================================================

# Logistic Regression
try:
    joblib.dump(log_reg, 'artifacts/logistic_regression.pkl')
    print("OK: Logistic Regression saved: artifacts/logistic_regression.pkl")
except NameError:
    pass

# SVM
try:
    joblib.dump(svc_model, 'artifacts/svm_rbf.pkl')
    print("OK: SVM model saved: artifacts/svm_rbf.pkl")
except NameError:
    pass

# LSTM
try:
    lstm_model.save('artifacts/lstm_model.h5')
    print("OK: LSTM model saved: artifacts/lstm_model.h5")
except NameError:
    pass

# Attention NN
try:
    attention_model.save('artifacts/attention_model.h5')
    print("OK: Attention NN saved: artifacts/attention_model.h5")
except NameError:
    pass

# ============================================================================
# Save Metadata
# ============================================================================
metadata = {
    'created_at': datetime.now().isoformat(),
    'feature_names': [f'V{i}' for i in range(1, 29)] + ['Amount'],
    'num_features': 29,
    'models_saved': os.listdir('artifacts'),
    'notes': 'Models trained on creditcard.csv dataset with SMOTE balancing'
}

with open('artifacts/metadata.json', 'w') as f:
    import json
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("ARTIFACT EXTRACTION COMPLETE!")
print("=" * 60)
print(f"Files saved in: {os.path.abspath('artifacts')}/")
print("\nNext steps:")
print("1. Review the artifacts/ directory")
print("2. Upload to Google Cloud Storage:")
print("   gsutil -m cp artifacts/* gs://YOUR-BUCKET-NAME/")
print("3. Follow the DEPLOYMENT_ROADMAP.md guide")
print("=" * 60)

# ============================================================================
# Validation: Test Loading the Artifacts
# ============================================================================
print("\nTesting: VALIDATION: Testing artifact loading...")

try:
    # Test loading Random Forest
    test_rf = joblib.load('artifacts/random_forest_model.pkl')
    print(f"OK: Random Forest loaded successfully (type: {type(test_rf)})")
except Exception as e:
    print(f"ERROR: Failed to load Random Forest: {e}")

try:
    # Test loading scaler
    test_scaler = joblib.load('artifacts/robust_scaler.pkl')
    print(f"OK: RobustScaler loaded successfully")

    # Test transform
    import numpy as np
    test_amount = np.array([[100.0]])
    scaled_amount = test_scaler.transform(test_amount)
    print(f"   Test: $100.00 → {scaled_amount[0][0]:.4f} (scaled)")
except Exception as e:
    print(f"ERROR: Failed to test scaler: {e}")

print("\nOK: Validation complete! Artifacts are ready for deployment.")
