"""
Unit tests for ML functions.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

def test_process_data_output_types(processed_data):
    """Test if process_data returns expected types."""
    # Check types
    assert isinstance(processed_data['X'], np.ndarray)
    assert isinstance(processed_data['y'], np.ndarray)
    assert isinstance(processed_data['encoder'], OneHotEncoder)
    assert isinstance(processed_data['lb'], LabelBinarizer)
    
    # Check shapes
    assert len(processed_data['X'].shape) == 2
    assert len(processed_data['y'].shape) == 1
    assert processed_data['X'].shape[0] == processed_data['y'].shape[0]

def test_train_model_output_types(trained_models):
    """Test if train_model returns expected model types."""
    # Check model types
    assert isinstance(trained_models['random_forest'], RandomForestClassifier)
    assert isinstance(trained_models['logistic_regression'], LogisticRegression)
    assert isinstance(trained_models['xgboost'], xgb.XGBClassifier)

def test_inference_output_type(processed_data, trained_models):
    """Test if inference returns expected type."""
    from src.ml.model import inference
    
    # Test inference for each model
    for model_type, model in trained_models.items():
        predictions = inference(model, processed_data['X'])
        
        # Check type and shape
        assert isinstance(predictions, np.ndarray)
        assert len(predictions.shape) == 1
        assert predictions.shape[0] == processed_data['X'].shape[0]

def test_process_data_inference_mode(processed_data, sample_data):
    """Test if process_data works correctly in inference mode."""
    from src.ml.data import process_data
    
    # Process in inference mode
    X_test, y_test, _, _ = process_data(
        sample_data,
        categorical_features=processed_data['categorical_features'],
        label="salary",
        training=False,
        encoder=processed_data['encoder'],
        lb=processed_data['lb']
    )
    
    # Check shapes match
    assert X_test.shape[1] == processed_data['X'].shape[1]
    assert y_test.shape == processed_data['y'].shape
