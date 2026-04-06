import pandas as pd

def preprocess(df):
    df = df.copy()

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Fill missing values
    df = df.fillna(0)

    # --- FEATURE ENGINEERING ---

    # Distance
    if "distance" not in df.columns:
        df["distance"] = 100  # fallback

    # Traffic
    if "traffic" in df.columns:
        df["traffic_score"] = df["traffic"]
    else:
        df["traffic_score"] = 0.5

    # Weather
    if "weather" in df.columns:
        df["weather_score"] = df["weather"].astype("category").cat.codes
    else:
        df["weather_score"] = 0

    # Transport mode encoding
    if "transport_mode" in df.columns:
        df["mode_encoded"] = df["transport_mode"].astype("category").cat.codes
    else:
        df["mode_encoded"] = 0

    # --- TARGET VARIABLE (DELAY) ---
    if "delay" in df.columns:
        df["target"] = df["delay"]
    elif "delivery_time" in df.columns and "expected_time" in df.columns:
        df["target"] = (df["delivery_time"] > df["expected_time"]).astype(int)
    else:
        df["target"] = 0  # fallback

    # Final feature set
    features = df[[
        "distance",
        "traffic_score",
        "weather_score",
        "mode_encoded"
    ]]

    target = df["target"]

    return features, target