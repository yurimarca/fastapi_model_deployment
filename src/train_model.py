"""
Author: Yuri Marca
Date: 2025-03-29

This script is used to train multiple machine learning models for census data classification.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ml import (
    RANDOM_SEED,
    logger,
    train_model,
    inference,
    compute_model_metrics,
    process_data,
    compute_slice_metrics,
    save_slice_metrics
)

# Start logging
logger.info("Starting model training process")

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the data
data_path = os.path.join(project_root, "data", "census.csv")
logger.info(f"Loading data from {data_path}")

# Check if the data file exists
if not os.path.exists(data_path):
    logger.error(f"Data file not found at {data_path}")
    raise FileNotFoundError(f"Data file not found at {data_path}")

# Load and clean the data
data = pd.read_csv(data_path)
# Clean column names by removing whitespace
data.columns = data.columns.str.strip()
logger.info(f"Loaded {len(data)} rows of data")
logger.info(f"Columns in dataset: {', '.join(data.columns)}")

# Split the data
logger.info(f"Splitting data with random seed {RANDOM_SEED}")
train, test = train_test_split(data, test_size=0.20, random_state=RANDOM_SEED)
logger.info(f"Training set size: {len(train)}, Test set size: {len(test)}")

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process training data
logger.info("Processing training data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
logger.info(f"Processed training data shape: {X_train.shape}")

# Process test data
logger.info("Processing test data")
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
logger.info(f"Processed test data shape: {X_test.shape}")

# Define model types to train
model_types = ['random_forest', 'logistic_regression', 'xgboost']

# Create model directory if it doesn't exist
model_dir = os.path.join(project_root, "artifacts")
os.makedirs(model_dir, exist_ok=True)

# Train and evaluate each model
for model_type in model_types:
    logger.info(f"*** Starting training process for {model_type} ***")
    
    # Train the model using the train_model function
    model = train_model(X_train, y_train, model_type=model_type)
    
    # Make predictions using the inference function
    y_pred = inference(model, X_test)
    
    # Compute metrics using compute_model_metrics
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    
    # Save the model
    model_path = os.path.join(model_dir, f"{model_type}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Compute and save slice metrics for a given feature
    feature = "education"
    logger.info(f"Computing slice metrics for {model_type} model")
    slice_metrics = compute_slice_metrics(
        model=model,
        data=test,
        feature=feature,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb
    )
    
    # Save slice metrics to file
    slice_output_path = os.path.join(model_dir, f"slice_output_{model_type}.txt")
    save_slice_metrics(slice_metrics, feature, slice_output_path)
    logger.info(f"Slice metrics saved to {slice_output_path}")

# Save the encoder and label binarizer
logger.info("Saving preprocessing components")
encoder_path = os.path.join(model_dir, "encoder.joblib")
lb_path = os.path.join(model_dir, "lb.joblib")
joblib.dump(encoder, encoder_path)
joblib.dump(lb, lb_path)
logger.info(f"Encoder saved to {encoder_path}")
logger.info(f"Label binarizer saved to {lb_path}")

logger.info("Model training process completed successfully")
