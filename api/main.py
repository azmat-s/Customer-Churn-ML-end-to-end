from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Load model once at startup
MODEL_PATH = Path(__file__).parent.parent / "models" / "lr_pipeline.joblib"
pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn based on their profile.",
    version="1.0.0"
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Define expected input schema
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


def calculate_engagement_score(data):
    # Tenure score (40%) - normalize to 0-100 (max tenure = 72)
    tenure_score = min(data['tenure'] / 72 * 100, 100)

    # Services score (30%) - count services used
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    services_used = sum(1 for s in services if data[s] == 'Yes')
    services_score = services_used / 6 * 100

    # Contract score (20%)
    contract_map = {'Month-to-month': 0, 'One year': 50, 'Two year': 100}
    contract_score = contract_map.get(data['Contract'], 0)

    # Payment score (10%)
    payment_map = {'Electronic check': 0, 'Mailed check': 33, 'Bank transfer (automatic)': 66, 'Credit card (automatic)': 100}
    payment_score = payment_map.get(data['PaymentMethod'], 0)

    return round(tenure_score * 0.4 + services_score * 0.3 + contract_score * 0.2 + payment_score * 0.1)


@app.post("/predict")
def predict(customer: CustomerInput):
    # Convert input to DataFrame (pipeline expects DataFrame with column names)
    input_df = pd.DataFrame([customer.model_dump()])

    # Get prediction and probability
    prediction = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    # Calculate engagement score
    score = calculate_engagement_score(customer.model_dump())

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": "High" if probability >= 0.5 else "Medium" if probability >= 0.3 else "Low",
        "engagement_score": score
    }