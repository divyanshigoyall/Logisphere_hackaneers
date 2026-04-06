import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def prepare_features(input_data):
    return [[
        input_data.get("distance", 100),
        input_data.get("traffic", 0.5),
        input_data.get("weather", 0),
        input_data.get("mode", 0)
    ]]

def predict_delay(input_data):
    features = prepare_features(input_data)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "delay": int(prediction),
        "delay_probability": float(probability)
    }