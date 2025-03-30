from fastapi import FastAPI
from mangum import Mangum

#app = FastAPI()
app = FastAPI(
    title="Test API",
    description="Simple API for testing",
    version="1.0.0",
    root_path="/prod"
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on AWS Lambda with Docker!"}

@app.post("/predict")
def predict(data: dict):
    return {"prediction": "some result"}

# AWS Lambda handler
handler = Mangum(app)
