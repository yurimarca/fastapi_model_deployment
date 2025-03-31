# Deploying a ML Model to Cloud Application Platform with FastAPI

This project is designed to deploy a machine learning model as a REST API on a cloud application platform using FastAPI. The repository follows a structured approach that meets the requirements outlined in the project rubric.

## Repository Structure

The repository is organized as follows:

```
.gitignore
CODEOWNERS
EDA.ipynb
LICENSE.txt
README.md
requirements.txt
setup.py
__pycache__/
.github/
    workflows/
.ipynb_checkpoints/
.pytest_cache/
api-container/
    Dockerfile
    copy_artifacts.sh
    requirements.txt
    app/         <-- Contains the FastAPI application code
artifacts/        <-- Contains serialized model artifacts and supporting files (encoder, etc.)
data/
logs/
model_cards/      <-- Contains the detailed model card documentation
screenshots/      <-- Contains the CI/CD and API documentation screenshots (e.g., continuous_integration.png, example.png, continuous_deployment.png, live_get.png, live_post.png)
src/              <-- Contains scripts for model training, inference and metrics computation
tests/            <-- Contains unit tests for ML functions and API endpoints
```

## Git and Continuous Integration

- **Git Setup & GitHub Actions:**  
  The project is version-controlled with Git. GitHub Actions is configured to run both `pytest` and `flake8` on every push to the `main/master` branch. You can view the CI workflow in the [`.github/workflows`](.github/workflows/) directory.

- **Continuous Integration Screenshot:**  
  A screenshot of the CI passing is provided as [`continuous_integration.png`](screenshots/continuous_integration.png). This confirms that all unit tests (at least 6) and lint checks pass.

## Model Building

- **Model Training:**  
  The model is trained on the provided dataset located in the `data/` directory. The training script (located in [src](src/)) splits the data using a train-test approach (or cross-validation) and calls the following functions:
  - `train_model()`: Trains the machine learning model.
  - `save_model()`: Persists the trained model and any categorical encoders.
  - `load_model()`: Loads the trained model and encoder for inference.
  - `inference()`: Handles model predictions.
  - `compute_metrics()`: Determines classification performance metrics.

- **Data Processing Script:**  
  A dedicated script processes the data, trains the model, and saves both the model and encoder. This script utilizes the functions above.

## Testing

- **Unit Tests:**  
  The project includes more than three unit tests under the [tests](tests/) directory. These tests ensure that functions return the correct types and expected outputs. They cover:
  - Model training and persistence.
  - Type output checks for inference functions.
  
- **Slice-Based Metrics:**  
  There is a function that computes performance metrics for slices of the data based on a fixed categorical variable (e.g., education). The function iterates through the unique values and prints metrics for each slice. The output is saved to `slice_output.txt`.

## Model Card

The project includes a comprehensive model card under the [model_cards](model_cards/) directory. The model card details:
- Model description and intended use.
- Data information and preprocessing steps.
- Training configuration.
- Detailed performance metrics including metrics computed on data slices.
- Limitations and ethical considerations.
  
The model card follows the provided template and is written in complete sentences.

## REST API

- **API Implementation:**  
  The REST API is built using FastAPI and is located in the [api-container/app](api-container/app/) directory. The API includes:
  - **GET /**: Returns a greeting message.
  - **POST /predict**: Accepts a JSON payload for model inference. The endpoint is implemented using Python type hints along with a Pydantic model which includes an example for the request body. FastAPI automatically generates interactive API documentation.

- **API Documentation Screenshot:**  
  A screenshot of the FastAPI docs showing the example is provided as [`example.png`](screenshots/example.png).

## API Deployment

- **Cloud Deployment with AWS:**  
  The application is containerized and deployed to AWS using the following services:
  - **Amazon Elastic Container Registry (ECR):** The Docker image is built and pushed to a private ECR repository.
  - **AWS Lambda:** The containerized FastAPI application is deployed as a Lambda function. This allows the API to run in a serverless environment.

  The deployment process is automated using GitHub Actions. The workflow:
  1. Builds the Docker image using the `api-container/Dockerfile`.
  2. Pushes the image to the ECR repository.
  3. Updates the Lambda function to use the latest image from ECR.

- **Deployment Screenshots:**  
  - [`continuous_deployment.png`](screenshots/continuous_deployment.png): Shows that continuous deployment is enabled.
  - [`live_get.png`](screenshots/live_get.png): A screenshot of the browser displaying the GET endpoint response.
  - [`live_post.png`](screenshots/live_post.png): A screenshot of the result and status code from the API POST request.

- **API Query Script:**  
  A Python script is included in the repository to query the API and verify its functionality. The script sends a POST request to the API and displays both the model inference result and the status code. You can find the script at [`query_api.py`](query_api.py).  