#!/usr/bin/env python3
"""Promote models/fraud_model.pkl.bak -> models/fraud_model.pkl

Run this from repository root (or from backend/) to copy the backup into place.
"""
import argparse
import os
import shutil
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
BAK = MODEL_DIR / "fraud_model.pkl.bak"
PKL = MODEL_DIR / "fraud_model.pkl"

parser = argparse.ArgumentParser()
parser.add_argument("--remove-bak", action="store_true", help="Remove the .bak after copying")
args = parser.parse_args()

if not BAK.exists():
    print(f"Backup file not found: {BAK}")
    raise SystemExit(2)

os.makedirs(MODEL_DIR, exist_ok=True)
try:
    # try atomic replace
    try:
        os.replace(str(BAK), str(PKL))
        print(f"Moved {BAK} -> {PKL}")
    except Exception:
        shutil.copyfile(str(BAK), str(PKL))
        print(f"Copied {BAK} -> {PKL}")
        if args.remove_bak:
            os.remove(str(BAK))
            print(f"Removed {BAK}")
except Exception as e:
    print("Failed to promote model:", e)
    raise

print("Promotion complete. Consider restarting the backend to reload the model.")
