"""
Shared test fixtures and configurations for the project.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'age': [25, 40],
        'workclass': ['Private', 'Public'],
        'fnlgt': [226802, 121772],
        'education': ['Bachelors', 'Masters'],
        'education-num': [13, 14],
        'marital-status': ['Never-married', 'Married'],
        'occupation': ['Tech-support', 'Admin'],
        'relationship': ['Not-in-family', 'Husband'],
        'race': ['White', 'Black'],
        'sex': ['Male', 'Female'],
        'capital-gain': [0, 0],
        'capital-loss': [0, 0],
        'hours-per-week': [40, 40],
        'native-country': ['United-States', 'United-States'],
        'salary': ['<=50K', '>50K']
    })

@pytest.fixture
def categorical_features():
    """Define categorical features for testing."""
    return [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]

@pytest.fixture
def numerical_features():
    """Define numerical features for testing."""
    return [
        'age', 'fnlgt', 'education-num', 'capital-gain',
        'capital-loss', 'hours-per-week'
    ]

@pytest.fixture
def encoder():
    """Create a fitted OneHotEncoder for testing."""
    return OneHotEncoder(sparse=False, handle_unknown='ignore')

@pytest.fixture
def lb():
    """Create a fitted LabelBinarizer for testing."""
    return LabelBinarizer()

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
    return X, y, encoder, lb

@pytest.fixture
def trained_models(processed_data):
    """Create trained models for testing."""
    from src.ml.model import train_model
    models = {}
    model_types = ['random_forest', 'logistic_regression', 'xgboost']
    
    for model_type in model_types:
        models[model_type] = train_model(
            processed_data[0],
            processed_data[1],
            model_type=model_type
        )
    
    return models

@pytest.fixture
def model_predictions(processed_data, trained_models):
    """Create model predictions for testing."""
    from src.ml.model import inference
    predictions = {}
    
    for model_type, model in trained_models.items():
        predictions[model_type] = inference(model, processed_data[0])
    
    return predictions 