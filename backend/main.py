from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from model_loader import load_model

model, encoder = load_model()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/fraud_model.pkl")

class UserData(BaseModel):
    age: int
    account_age: int
    total_transactions: int
    past_fraud: int
    avg_order_value: float

@app.post("/predict_user")
def predict_user(data: UserData):
    features = np.array([[data.age, data.account_age, data.total_transactions,
                        data.past_fraud, data.avg_order_value]])
    prob = model.predict_proba(features)[0][1] * 100
    risk = "High" if prob > 70 else "Medium" if prob > 40 else "Low"
    return {"fraud_probability": round(prob, 2), "risk_level": risk}

class TransactionData(BaseModel):
    amount: float
    payment_method: str
    device: str
    ip: str | None
    browser: str | None
    shipping: str
    billing: str | None

@app.post("/predict_transaction")
def predict_transaction(data: TransactionData):
    risk_points = 0
    if data.amount > 500: risk_points += 30
    if data.amount > 2000: risk_points += 20
    if data.payment_method in ["Credit Card","PayPal"]: risk_points += 10
    if data.billing and data.billing != data.shipping: risk_points += 20
    if data.device == "Mobile": risk_points += 10

    features = np.array([[data.amount,0,0,0,0]])
    base_prob = model.predict_proba(features)[0][1] * 100
    final_prob = min(100, base_prob + risk_points)

    risk = "High" if final_prob > 75 else "Medium" if final_prob > 40 else "Low"
    return {"fraud_probability": round(final_prob,2),"risk_level":risk}

@app.get("/")
def home():
    return {"message":"FraudGuard API running"}
