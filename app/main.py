from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# Load model at startup
model = joblib.load("model.pkl")

class Patient(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Patient):
    X = np.array([[ 
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    proba = model.predict_proba(X)[0]
    pred = int(np.argmax(proba))

    return {
        "prediction": pred,
        "confidence": float(max(proba))
    }

