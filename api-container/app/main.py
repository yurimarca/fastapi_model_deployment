from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from mangum import Mangum
import joblib
import numpy as np
from pathlib import Path
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Census Income Prediction API",
    description="API for predicting income based on census data",
    version="1.0.0"
)

# Define the Pydantic model for input data
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
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
        }

# Load the model and preprocessing components
MODEL_PATH = Path(__file__).parent / "artifacts" / "xgboost.joblib"
ENCODER_PATH = Path(__file__).parent / "artifacts" / "encoder.joblib"
LB_PATH = Path(__file__).parent / "artifacts" / "lb.joblib"

logger.info(f"Loading model from: {MODEL_PATH}")
logger.info(f"Loading encoder from: {ENCODER_PATH}")
logger.info(f"Loading label binarizer from: {LB_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    lb = joblib.load(LB_PATH)
    logger.info("Successfully loaded model and preprocessing components")
except Exception as e:
    logger.error(f"Error loading model or preprocessing components: {e}")
    model = None
    encoder = None
    lb = None

@app.get("/")
async def read_root():
    """
    Root endpoint returning a welcome message.
    """
    return {
        "message": "Welcome to the Census Income Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "root": "GET /",
            "predict": "POST /predict"
        }
    }

@app.post("/predict")
async def predict(data: CensusData):
    """
    Endpoint for making predictions using the trained model.
    
    Args:
        data (CensusData): Input data for prediction
        
    Returns:
        dict: Prediction result and probability
    """
    if model is None or encoder is None or lb is None:
        raise HTTPException(
            status_code=500,
            detail="Model or preprocessing components not loaded"
        )
    
    try:
        # Convert input data to DataFrame
        input_data = data.dict(by_alias=True)
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        X = encoder.transform(input_df)
        
        # Make prediction
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        
        # Convert prediction back to original label
        prediction_label = lb.inverse_transform(prediction)[0]
        
        return {
            "prediction": prediction_label,
            "probability": float(probability[0][1]),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

# AWS Lambda handler
handler = Mangum(app)
