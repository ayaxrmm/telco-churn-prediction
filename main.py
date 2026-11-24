from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = "model.joblib"

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, description="Monthly bill, USD")
    TotalCharges: float = Field(..., ge=0, description="Total amount paid, USD")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.5,
                "TotalCharges": 191.0,
            }
        }
    }


class Prediction(BaseModel):
    """То, что API возвращает в ответ."""
    churn: bool
    probability: float
    risk: Literal["low", "medium", "high"]


def risk_level(probability: float) -> str:
    if probability < 0.3:
        return "low"
    if probability < 0.6:
        return "medium"
    return "high"


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(df)[0, 1])
    return Prediction(
        churn=probability >= 0.5,
        probability=round(probability, 3),
        risk=risk_level(probability),
    )
