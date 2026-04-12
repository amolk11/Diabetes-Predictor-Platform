from fastapi import FastAPI
import numpy as np

from deployment.schemas import PatientData
from deployment.model_loader import load_model_and_scaler

# Load model
model, scaler = load_model_and_scaler()

app = FastAPI(title="Diabetes Prediction API")

@app.get("/")
def home():
    return {"message": "API Running 🚀"}

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([[ 
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }