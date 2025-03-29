"""
Author: Yuri Marca
Date: 2025-03-29

This module contains the code for training and evaluating machine learning models.
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from . import RANDOM_SEED, logger
import pandas as pd
import numpy as np
from .data import process_data


def train_model(X_train, y_train, model_type='random_forest'):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    model_type : str
        Type of model to train ('random_forest', 'logistic_regression', or 'xgboost')
    Returns
    -------
    model
        Trained machine learning model.
    """
    logger.info(f"Training {model_type} model with random seed {RANDOM_SEED}")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=RANDOM_SEED)
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(random_state=RANDOM_SEED)
    else:
        logger.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    logger.info(f"Successfully trained {model_type} model")
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    
    logger.info(f"Model metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta: {fbeta:.4f}")
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : trained model object
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logger.info("Running model inference")
    preds = model.predict(X)
    logger.info(f"Generated {len(preds)} predictions")
    return preds


def compute_slice_metrics(model, data, feature, categorical_features, encoder, lb):
    """
    Compute performance metrics for each slice of data based on a categorical feature.

    Inputs
    ------
    model : trained model object
        Trained machine learning model.
    data : pd.DataFrame
        Raw data containing the feature to slice on.
    feature : str
        Name of the categorical feature to slice on.
    categorical_features : list
        List of categorical features used in training.
    encoder : OneHotEncoder
        Fitted OneHotEncoder used in training.
    lb : LabelBinarizer
        Fitted LabelBinarizer used in training.

    Returns
    -------
    dict
        Dictionary containing metrics for each slice.
    """
    if feature not in categorical_features:
        raise ValueError(f"Feature {feature} not found in categorical features")

    # Get unique values for the feature
    unique_values = data[feature].unique()
    slice_metrics = {}

    for value in unique_values:
        # Create slice of data
        slice_data = data[data[feature] == value].copy()
        
        # Process the slice
        X_slice, y_slice, _, _ = process_data(
            slice_data,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )
        
        # Get predictions
        y_pred = inference(model, X_slice)
        
        # Compute metrics
        precision, recall, fbeta = compute_model_metrics(y_slice, y_pred)
        
        slice_metrics[value] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta,
            'sample_size': len(slice_data)
        }

    return slice_metrics


def save_slice_metrics(slice_metrics, feature, output_file):
    """
    Save slice metrics to a file.

    Inputs
    ------
    slice_metrics : dict
        Dictionary containing metrics for each slice.
    feature : str
        Name of the feature that was sliced.
    output_file : str
        Path to the output file.
    """
    with open(output_file, 'w') as f:
        f.write(f"Performance metrics for slices of feature: {feature}\n")
        f.write("=" * 50 + "\n\n")
        
        for value, metrics in slice_metrics.items():
            f.write(f"Slice: {value}\n")
            f.write(f"Sample size: {metrics['sample_size']}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F-beta: {metrics['fbeta']:.4f}\n")
            f.write("-" * 30 + "\n")
