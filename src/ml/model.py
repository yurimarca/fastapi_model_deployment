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
