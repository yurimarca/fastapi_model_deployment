import requests
import json

# Replace with your API endpoint
API_URL = "https://9qj8rsark0.execute-api.us-east-1.amazonaws.com/predict"

# Example payload
payload = {
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

# Send POST request
response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

# Display response
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")
