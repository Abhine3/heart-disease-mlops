from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import time
from fastapi import Request


app = FastAPI(title="Heart Disease Prediction API")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

request_count = 0
#--------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count
    start_time = time.time()

    response = await call_next(request)

    duration = round(time.time() - start_time, 4)
    request_count += 1

    logging.info(
        f"method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"latency={duration}s "
        f"total_requests={request_count}"
    )

    return response


#-------------------------------------------------------
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

