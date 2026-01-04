import pandas as pd
import joblib
import os

def test_model_file_exists():
    """
    Ensure model artifact is present.
    """
    assert os.path.exists("model.pkl"), "model.pkl not found"


def test_model_prediction():
    """
    Test model can load and predict.
    """
    model = joblib.load("model.pkl")

    sample = pd.DataFrame([{
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }])

    pred = model.predict(sample)
    assert pred[0] in [0, 1]

