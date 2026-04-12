from pydantic import BaseModel, Field

class PatientData(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies (0–20)")
    
    Glucose: float = Field(..., ge=0, le=300, description="Glucose level (0–300 mg/dL)")
    
    BloodPressure: float = Field(..., ge=0, le=200, description="Blood pressure (0–200 mm Hg)")
    
    SkinThickness: float = Field(..., ge=0, le=100, description="Skin thickness (0–100 mm)")
    
    Insulin: float = Field(..., ge=0, le=900, description="Insulin level (0–900 µU/mL)")
    
    BMI: float = Field(..., ge=10, le=70, description="Body Mass Index (10–70)")
    
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, le=3.0, description="DPF (0–3)")
    
    Age: int = Field(..., ge=1, le=120, description="Age (1–120 years)")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No Diabetes, 1 = Diabetes")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0–1)")