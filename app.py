from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
import os

# Model path
MODEL_PATH = "model.joblib"

# Load model and preprocessing objects
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

model, scaler, pca = load_model()

# Define FastAPI app
app = FastAPI()

# Define input schema
class ChurnRequest(BaseModel):
    account_length: int
    area_code: int
    international_plan: str
    voice_mail_plan: str
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    customer_service_calls: int

# Preprocessing function
def preprocess(data: ChurnRequest):
    try:
        # Convert categorical variables
        international_plan = 1 if data.international_plan.strip().lower() == "yes" else 0
        voice_mail_plan = 1 if data.voice_mail_plan.strip().lower() == "yes" else 0
        
        # Prepare input array
        features = np.array([
            data.account_length, data.area_code, international_plan,
            voice_mail_plan, data.number_vmail_messages, data.total_day_minutes,
            data.total_day_calls, data.total_day_charge, data.total_eve_minutes,
            data.total_eve_calls, data.total_eve_charge, data.total_night_minutes,
            data.total_night_calls, data.total_night_charge, data.total_intl_minutes,
            data.total_intl_calls, data.total_intl_charge, data.customer_service_calls
        ]).reshape(1, -1)
        
        # Apply scaling
        features = scaler.transform(features)
        
        # Apply PCA if available
        if pca:
            features = pca.transform(features)
        
        return features
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {e}")

# Prediction route
@app.post("/predict")
def predict(request: ChurnRequest):
    try:
        # Preprocess input
        processed_features = preprocess(request)
        
        # Make prediction
        prediction = model.predict(processed_features)
        
        return {"churn": bool(prediction[0])}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Retrain route (reload model from disk)
@app.post("/retrain")
def retrain():
    global model, scaler, pca
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail="Model file not found.")
        
        model, scaler, pca = load_model()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")

