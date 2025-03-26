from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Define paths
MODEL_PATH = Path("C:/Users/LEGION/OneDrive/Desktop/dev_project/predictree-ward-visit-main/predictree-ward-visit-main/backend/models/rf_tuned_model.pkl")
LABEL_ENCODERS_PATH = Path("C:/Users/LEGION/OneDrive/Desktop/dev_project/predictree-ward-visit-main/predictree-ward-visit-main/backend/models/label_encoders.pkl")
LABEL_ENCODERS_2_PATH = Path("C:/Users/LEGION/OneDrive/Desktop/dev_project/predictree-ward-visit-main/predictree-ward-visit-main/backend/models/label_encoders_2.pkl")
FEATURE_COLUMNS_PATH = Path("C:/Users/LEGION/OneDrive/Desktop/dev_project/predictree-ward-visit-main/predictree-ward-visit-main/backend/models/feature_columns.pkl")

# Load model and encoders
def load_model_and_encoders():
    try:
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        label_encoders_2 = joblib.load(LABEL_ENCODERS_2_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        return model, label_encoders, label_encoders_2, feature_columns
    except Exception as e:
        raise RuntimeError(f"Error loading model or encoders: {str(e)}")

# FastAPI app setup
app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class PatientData(BaseModel):
    age: int
    time_in_hospital: int
    n_lab_procedures: int
    n_procedures: int
    n_medications: int
    glucose_test: str
    a1c_test: str
    change_medication: str
    diabetes_medication: str

# Preprocessing function
def preprocess_input(data: dict, label_encoders, label_encoders_2):
    df = pd.DataFrame([data])
    
    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    for col, le in label_encoders_2.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    return df

# Prediction endpoint
@app.post("/predict")
def predict_readmission(patient: PatientData):
    try:
        model, label_encoders, label_encoders_2, feature_columns = load_model_and_encoders()
        processed_data = preprocess_input(patient.dict(), label_encoders, label_encoders_2)

        missing_features = set(feature_columns) - set(processed_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        prediction = model.predict(processed_data[feature_columns])[0]
        probability = model.predict_proba(processed_data[feature_columns])[0]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability[1]),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
