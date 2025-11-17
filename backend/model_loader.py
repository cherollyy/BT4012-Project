# backend/model_loader.py
import joblib
import os

MODEL_PATH = "models/fraud_model.pkl"
ENCODER_PATH = "models/encoder.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None
    return model, encoder
