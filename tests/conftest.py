"""
Shared test fixtures and configurations for the project.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a small sample dataset
    data = {
        'age': [25, 30, 35, 40],
        'workclass': ['Private', 'Public', 'Private', 'Public'],
        'education': ['Bachelors', 'Masters', 'Bachelors', 'Masters'],
        'marital-status': ['Single', 'Married', 'Single', 'Married'],
        'occupation': ['Tech', 'Admin', 'Tech', 'Admin'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband'],
        'race': ['White', 'Black', 'White', 'Black'],
        'sex': ['Male', 'Female', 'Male', 'Female'],
        'native-country': ['US', 'UK', 'US', 'UK'],
        'salary': ['<=50K', '>50K', '<=50K', '>50K']
    }
    return pd.DataFrame(data)

@pytest.fixture
def categorical_features():
    """Define categorical features for testing."""
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

@pytest.fixture
def processed_data(sample_data, categorical_features):
    """Create processed data for testing."""
    from src.ml.data import process_data
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    return {
        'X': X,
        'y': y,
        'encoder': encoder,
        'lb': lb,
        'categorical_features': categorical_features
    }

@pytest.fixture
def trained_models(processed_data):
    """Create trained models for testing."""
    from src.ml.model import train_model
    models = {}
    model_types = ['random_forest', 'logistic_regression', 'xgboost']
    
    for model_type in model_types:
        models[model_type] = train_model(
            processed_data['X'],
            processed_data['y'],
            model_type=model_type
        )
    
    return models

@pytest.fixture
def model_predictions(processed_data, trained_models):
    """Create model predictions for testing."""
    from src.ml.model import inference
    predictions = {}
    
    for model_type, model in trained_models.items():
        predictions[model_type] = inference(model, processed_data['X'])
    
    return predictions 