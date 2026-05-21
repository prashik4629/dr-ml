from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.core.security import verify_api_key
from backend.services.predictor import predict


router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"],
)


class DiabetesInput(BaseModel):
    Pregnancies:             int   = Field(..., ge=0,   le=20,  example=2)
    Glucose:                 float = Field(..., ge=0,   le=400, example=120)
    BloodPressure:           float = Field(..., ge=0,   le=250, example=70)
    SkinThickness:           float = Field(..., ge=0,   le=100, example=25)
    Insulin:                 float = Field(..., ge=0,   le=900, example=80)
    BMI:                     float = Field(..., ge=0.0, le=70.0,example=28.5)
    DiabetesPedigreeFunction:float = Field(..., ge=0.0, le=3.0, example=0.45)
    Age:                     int   = Field(..., ge=1,   le=120, example=35)


class DiabetesOutput(BaseModel):
    prediction:  int
    probability: float
    diagnosis:   str
    risk_level:  str


@router.post(
    "/predict",
    response_model=DiabetesOutput,
    summary="Predict diabetes risk",
)
async def predict_diabetes(
    input_data: DiabetesInput,
    api_key: str = Depends(verify_api_key),
):
    result = predict(input_data.model_dump())

    prediction  = result["prediction"]
    probability = result["probability"]

    if probability >= 0.75:
        risk_level = "High Risk"
    elif probability >= 0.45:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    return DiabetesOutput(
        prediction=prediction,
        probability=probability,
        diagnosis=(
            "Diabetes Detected"
            if prediction == 1
            else "No Diabetes Detected"
        ),
        risk_level=risk_level,
    )