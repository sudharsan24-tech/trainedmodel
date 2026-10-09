
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from diabetes_predictor import predict_diabetes

app = FastAPI(
    title="SilentSigns Diabetes Screening API",
    description="Experimental diabetes risk screening endpoint",
    version="1.0.0"
)


class PatientData(BaseModel):
    Age: int = Field(ge=1, le=120)
    Gender: int = Field(ge=0, le=1)

    Polyuria: int = Field(ge=0, le=1)
    Polydipsia: int = Field(ge=0, le=1)
    sudden_weight_loss: int = Field(ge=0, le=1)
    weakness: int = Field(ge=0, le=1)
    Polyphagia: int = Field(ge=0, le=1)
    Genital_thrush: int = Field(ge=0, le=1)
    visual_blurring: int = Field(ge=0, le=1)
    Itching: int = Field(ge=0, le=1)
    Irritability: int = Field(ge=0, le=1)
    delayed_healing: int = Field(ge=0, le=1)
    partial_paresis: int = Field(ge=0, le=1)
    muscle_stiffness: int = Field(ge=0, le=1)
    Alopecia: int = Field(ge=0, le=1)
    Obesity: int = Field(ge=0, le=1)


@app.get("/")
def home():
    return {
        "message": "SilentSigns Diabetes Screening API is running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PatientData):
    # Convert API field names to the model's exact feature names
    patient_data = {
        "Age": data.Age,
        "Gender": data.Gender,
        "Polyuria": data.Polyuria,
        "Polydipsia": data.Polydipsia,
        "sudden weight loss": data.sudden_weight_loss,
        "weakness": data.weakness,
        "Polyphagia": data.Polyphagia,
        "Genital thrush": data.Genital_thrush,
        "visual blurring": data.visual_blurring,
        "Itching": data.Itching,
        "Irritability": data.Irritability,
        "delayed healing": data.delayed_healing,
        "partial paresis": data.partial_paresis,
        "muscle stiffness": data.muscle_stiffness,
        "Alopecia": data.Alopecia,
        "Obesity": data.Obesity
    }

    try:
        return predict_diabetes(patient_data)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the model files and server logs."
        ) from exc
