# backend/preprocessing.py
import pandas as pd

def preprocess_user(data: dict, encoder=None):
    df = pd.DataFrame([data])

    # Example: future categorical encoding
    if encoder:
        df = encoder.transform(df)

    return df.values  # return numpy array


def preprocess_transaction(data: dict, encoder=None):
    df = pd.DataFrame([data])

    # Example: convert categorical columns later
    if encoder:
        df = encoder.transform(df)

    return df.values
