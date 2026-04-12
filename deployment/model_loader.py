import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_model_and_scaler():
    model_path = BASE_DIR / "models/trained/best_model.pkl"
    scaler_path = BASE_DIR / "models/scaler/scaler.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler