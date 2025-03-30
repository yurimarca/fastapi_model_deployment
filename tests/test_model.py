"""
Unit tests for ML functions.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from src.ml.data import process_data
from src.ml.model import train_model, inference, compute_model_metrics

def test_process_data_output_types(processed_data):
    """Test that process_data returns correct types."""
    X, y, encoder, lb = processed_data
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)

def test_inference_output_type(processed_data, trained_models):
    """Test that inference returns correct type."""
    X, y, _, _ = processed_data
    model = trained_models['random_forest']
    predictions = inference(model, X)
    assert isinstance(predictions, np.ndarray)

def test_process_data_inference_mode(sample_data, categorical_features):
    """Test process_data in inference mode."""
    # First fit the encoder and lb on training data
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    
    # Then use the fitted components for inference
    X_inf, y_inf, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    assert isinstance(X_inf, np.ndarray)
    assert isinstance(y_inf, np.ndarray)
    assert X_inf.shape[1] == X_train.shape[1]  # Same number of features

def test_train_model_output_types(trained_models):
    """Test that train_model returns correct model types."""
    assert 'random_forest' in trained_models
    assert 'logistic_regression' in trained_models
    assert 'xgboost' in trained_models
    
    for model in trained_models.values():
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

def test_compute_model_metrics(model_predictions, processed_data):
    """Test that compute_model_metrics returns correct types and ranges."""
    _, y_true, _, _ = processed_data
    predictions = model_predictions['random_forest']
    
    precision, recall, fbeta = compute_model_metrics(y_true, predictions)
    
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
