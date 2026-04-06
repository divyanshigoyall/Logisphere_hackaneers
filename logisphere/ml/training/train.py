from load_data import load_data
from preprocess import preprocess
from sklearn.ensemble import RandomForestClassifier
import pickle

MODEL_PATH = "../models/model.pkl"

def train():
    df = load_data()

    X, y = preprocess(df)

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved as model.pkl")

if __name__ == "__main__":
    train()