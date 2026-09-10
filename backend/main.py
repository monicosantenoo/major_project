from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

import pandas as pd
import joblib
import os


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Machine Learning API for Heart Disease Prediction",
    version="1.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "heart_disease_model.pkl"
)

FRONTEND_PATH = os.path.join(
    BASE_DIR,
    "frontend"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# SERVE FRONTEND FILES
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_PATH),
    name="static"
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return FileResponse(
        os.path.join(FRONTEND_PATH, "index.html")
    )


# --------------------------------------------------
# INPUT DATA
# --------------------------------------------------

class PatientData(BaseModel):

    age: float
    sex: str
    dataset: str
    cp: str
    trestbps: float
    chol: float
    fbs: str
    restecg: str
    thalch: float
    exang: str
    oldpeak: float
    slope: str
    ca: float
    thal: str


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

@app.post("/predict")
def predict(data: PatientData):

    patient = pd.DataFrame([{

        "age": data.age,
        "sex": data.sex,
        "dataset": data.dataset,
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalch": data.thalch,
        "exang": data.exang,
        "oldpeak": data.oldpeak,
        "slope": data.slope,
        "ca": data.ca,
        "thal": data.thal

    }])

    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0][1]

    if prediction == 1:
        result = "Heart Disease Present"
    else:
        result = "No Heart Disease"

    return {
        "prediction": int(prediction),
        "result": result,
        "probability": round(
            float(probability) * 100,
            2
        )
    }