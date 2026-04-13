from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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
app = FastAPI(
    title="Diabetes Prediction API",
    description="ML API for predicting diabetes risk",
    version="1.0.0"
)


# -----------------------------
# CORS (VERY IMPORTANT for frontend)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change later to specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Global exception handler
# -----------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal Server Error",
            "details": str(exc)
        }
    )


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "diabetes-api"
    }


# -----------------------------
# Home route
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Diabetes Prediction API is running 🚀"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    start_time = time.time()

    try:
        # -----------------------------
        # Prepare input
        # -----------------------------
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

        # -----------------------------
        # Preprocess
        # -----------------------------
        input_scaled = scaler.transform(input_data)

        # -----------------------------
        # Predict
        # -----------------------------
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])

        # -----------------------------
        # Risk Label (bonus)
        # -----------------------------
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # -----------------------------
        # Latency
        # -----------------------------
        latency = round((time.time() - start_time) * 1000, 2)

        # -----------------------------
        # Response
        # -----------------------------
        response = {
            "status": "success",
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "latency_ms": latency,
            "input_data": {
                "Pregnancies": data.Pregnancies,
                "Glucose": data.Glucose,
                "BloodPressure": data.BloodPressure,
                "SkinThickness": data.SkinThickness,
                "Insulin": data.Insulin,
                "BMI": data.BMI,
                "DiabetesPedigreeFunction": data.DiabetesPedigreeFunction,
                "Age": data.Age
            }
        }

        logger.info(f"Prediction successful | latency={latency}ms")

        return response


    # -----------------------------
    # Validation error
    # -----------------------------
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )

    # -----------------------------
    # General error
    # -----------------------------
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )