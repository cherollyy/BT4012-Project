# backend/training/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import json
import os
from pathlib import Path

# prefer an explicit env var, else look for a repo-local data file
RAW_DATA_PATH = os.getenv(
    "RAW_DATA_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "Fraudulent_E-Commerce_Transaction_Data.csv")
)

raw_path = Path(RAW_DATA_PATH)
if not raw_path.exists():
    raise FileNotFoundError(
        f"Training CSV not found. Looked for RAW_DATA_PATH={raw_path}.\n\n"
        "Place the raw CSV at data/ or set RAW_DATA_PATH environment variable."
    )

df = pd.read_csv(raw_path)
# 1) Feature engineering step
# TODO: Add more engineered features here

X = df.drop("fraud", axis=1)
y = df["fraud"]

# 2) Encode categoricals
categorical_cols = X.select_dtypes(include="object").columns
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(X[categorical_cols])

X_encoded = encoder.transform(X[categorical_cols]).toarray()
X_final = X_encoded  # later you can add numeric columns too

# 3) Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_final, y)

# 4) Save model + encoder + version tag
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/fraud_model.pkl")
joblib.dump(encoder, "../models/encoder.pkl")

with open("../models/model_version.txt", "w") as f:
    f.write("Version: 1.0")
