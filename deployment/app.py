from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import numpy as np
import logging
import time

from deployment.schemas import PatientData, PredictionResponse
from deployment.model_loader import load_model_and_scaler

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# -----------------------------
# Load model
# -----------------------------
try:
    model, scaler = load_model_and_scaler()
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model loading failed")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Diabetes Prediction API")

# -----------------------------
# Global exception handler
# -----------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": str(exc)}
    )

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Home
# -----------------------------
@app.get("/")
def home():
    return {"message": "API Running 🚀"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    start_time = time.time()

    try:
        # Convert input to array
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

        # Validate shape
        if input_data.shape != (1, 8):
            raise ValueError("Invalid input shape")

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        response = {
            "prediction": int(prediction),
            "probability": float(probability)
        }

        # Logging
        latency = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Prediction: {response}, latency={latency}ms")

        return response

    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")