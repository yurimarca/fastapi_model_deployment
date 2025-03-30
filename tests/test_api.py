"""
Unit tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add the api-container directory to the Python path
api_container_path = Path(__file__).parent.parent / "api-container"
sys.path.append(str(api_container_path))

from app.main import app

# Mock the model and preprocessing components
@pytest.fixture
def mock_model():
    with patch('app.main.model') as mock:
        mock.predict.return_value = np.array([1])
        mock.predict_proba.return_value = np.array([[0.2, 0.8]])
        yield mock

@pytest.fixture
def mock_encoder():
    with patch('app.main.encoder') as mock:
        mock.transform.return_value = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        yield mock

@pytest.fixture
def mock_lb():
    with patch('app.main.lb') as mock:
        mock.inverse_transform.return_value = np.array(['>50K'])
        yield mock

@pytest.fixture
def test_client(mock_model, mock_encoder, mock_lb):
    return TestClient(app)

def test_get_root(test_client):
    """
    Test the GET endpoint at root.
    """
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data
    assert data["message"] == "Welcome to the Census Income Prediction API"
    assert data["version"] == "1.0.0"
    assert "root" in data["endpoints"]
    assert "predict" in data["endpoints"]

def test_post_predict_success(test_client):
    """
    Test the POST endpoint for successful prediction.
    """
    test_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    response = test_client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "status" in data
    assert data["status"] == "success"
    assert isinstance(data["probability"], float)
    assert 0 <= data["probability"] <= 1
    assert data["prediction"] == ">50K"

def test_post_predict_invalid_data(test_client):
    """
    Test the POST endpoint with invalid data.
    """
    test_data = {
        "age": "invalid",  # Age should be an integer
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    response = test_client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error
