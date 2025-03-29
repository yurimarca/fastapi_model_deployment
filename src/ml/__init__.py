"""
ML module for model training and inference.
"""

import logging
import os
from datetime import datetime

# Global random seed for reproducibility
RANDOM_SEED = 42

def setup_logging():
    """
    Set up logging configuration for the ML module.
    Creates a log directory if it doesn't exist and configures logging to both file and console.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model_training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Create a default logger instance
logger = setup_logging()

# Import and expose the main functions
from .model import train_model, inference, compute_model_metrics
from .data import process_data

__all__ = [
    'RANDOM_SEED',
    'logger',
    'setup_logging',
    'train_model',
    'inference',
    'compute_model_metrics',
    'process_data'
]
