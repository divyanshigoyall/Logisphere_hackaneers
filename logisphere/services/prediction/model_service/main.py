from fastapi import FastAPI
from predict import predict_delay

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Prediction Service Running"}

@app.post("/predict")
def predict(data: dict):
    return predict_delay(data)