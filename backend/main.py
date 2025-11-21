from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from model_loader import load_model
import json
import math
import pandas as pd
import re
import importlib.util
from pathlib import Path
import traceback
import shutil
from fastapi import Header
import os
import threading
import tempfile
import time
import logging

model, encoder, metadata = load_model()
# training state
training_in_progress = False
# training progress info shared with frontend
training_progress = {
    "in_progress": False,
    "percent": 0,
    "message": None,
    "start_time": None,
}

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EnsemblePredictor:
    """A thin ensemble wrapper that combines two estimators with weights.

    Expects objects with a predict_proba(X) -> array-like (n_samples, 2)
    """
    def __init__(self, lgbm, xgb, w_lgbm=0.6, w_xgb=0.4):
        self.lgbm = lgbm
        self.xgb = xgb
        self.w_lgbm = w_lgbm
        self.w_xgb = w_xgb

    def predict_proba(self, X):
        p1 = self.lgbm.predict_proba(X)[:, 1]
        p2 = self.xgb.predict_proba(X)[:, 1]
        combined = (self.w_lgbm * p1) + (self.w_xgb * p2)
        # return as two-column array to mimic sklearn API
        return np.vstack([1 - combined, combined]).T


def _load_and_run_bt4012_model():
    """Dynamically import and run the training function from backend/models/bt4012_model.py
    without modifying that file.
    Returns the model_bundle dict returned by the script.
    """
    module_path = Path(__file__).resolve().parents[0] / "models" / "bt4012_model.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Model training file not found: {module_path}")

    spec = importlib.util.spec_from_file_location("bt4012_model", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # re-raise with traceback for API response
        raise

    if not hasattr(mod, "run_high_perf_tree_ensemble_model_only"):
        raise AttributeError("bt4012_model module does not expose run_high_perf_tree_ensemble_model_only")

    # run the training; if the training function supports a progress_callback
    # we pass one to receive incremental updates.
    def progress_cb(pct, msg=None):
        try:
            training_progress["in_progress"] = True
            training_progress["percent"] = int(max(0, min(100, int(pct))))
            training_progress["message"] = str(msg) if msg is not None else None
        except Exception:
            pass

    # prefer to call with progress callback if the function accepts it
    if hasattr(mod, "run_high_perf_tree_ensemble_model_only"):
        func = getattr(mod, "run_high_perf_tree_ensemble_model_only")
        try:
            bundle = func(random_state=42, progress_callback=progress_cb)
        except TypeError:
            # function doesn't accept progress_callback
            bundle = func(random_state=42)
    else:
        raise AttributeError("bt4012_model module does not expose run_high_perf_tree_ensemble_model_only")
    return bundle


def _save_model_atomic(model_obj, dest_path: Path):
    """Save a model object to dest_path atomically using a temp file + replace."""
    dest_dir = dest_path.parent
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest_dir), prefix="tmp_model_", suffix=".pkl")
    os.close(fd)
    try:
        joblib.dump(model_obj, tmp)
        # atomic replace
        os.replace(tmp, str(dest_path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _background_train_and_load():
    """Background training worker: runs training, saves model atomically, writes metadata, and updates globals."""
    global model, encoder, metadata, training_in_progress
    if training_in_progress:
        logger.info("Training already in progress; skipping background start.")
        return
    training_in_progress = True
    logger.info("Background training started.")
    try:
        try:
            training_progress["in_progress"] = True
            training_progress["percent"] = 1
            training_progress["message"] = "background training started"
            training_progress["start_time"] = time.time()
        except Exception:
            pass

        bundle = _load_and_run_bt4012_model()
    except Exception:
        logger.exception("Training failed during background run")
        training_progress["in_progress"] = False
        training_progress["message"] = "error during training"
        training_in_progress = False
        return

    # Build model_obj if necessary
    model_obj = bundle.get("model")
    if model_obj is None:
        lgbm = bundle.get("lgbm")
        xgb = bundle.get("xgb")
        if lgbm is None or xgb is None:
            logger.error("Training finished but returned no models")
            training_in_progress = False
            return
        ew = bundle.get("ensemble_weights") or {"lgbm": 0.6, "xgb": 0.4}
        w_lgbm = float(ew.get("lgbm", 0.6))
        w_xgb = float(ew.get("xgb", 0.4))
        model_obj = EnsemblePredictor(lgbm, xgb, w_lgbm=w_lgbm, w_xgb=w_xgb)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fraud_model.pkl"

    try:
        _save_model_atomic(model_obj, model_path)
        # save metadata atomically
        meta = {
            "feature_names": bundle.get("feature_names", []),
            "ensemble_weights": bundle.get("ensemble_weights", {}),
            "best_threshold": bundle.get("best_threshold"),
        }
        mf_tmp = model_dir / "metadata.json.tmp"
        with open(mf_tmp, "w") as f:
            json.dump(meta, f)
        os.replace(str(mf_tmp), str(model_dir / "metadata.json"))

        # update in-memory model
        model = model_obj
        metadata = meta
        logger.info(f"Background training complete, model saved to {model_path}")
        try:
            training_progress["in_progress"] = False
            training_progress["percent"] = 100
            training_progress["message"] = "completed"
        except Exception:
            pass
    except Exception:
        logger.exception("Failed to save model after training")
    finally:
        training_in_progress = False


@app.on_event("startup")
def startup_check_and_train():
    """On startup, if no model is loaded, start background training and return immediately."""
    global model, training_in_progress
    # if model already loaded, nothing to do
    if model is not None:
        logger.info("Model already loaded on startup")
        return

    # Optionally control via env var TRAIN_ON_STARTUP (default true)
    tos = os.getenv("TRAIN_ON_STARTUP", "1")
    if tos in ("0", "false", "False"):
        logger.info("TRAIN_ON_STARTUP disabled; skipping automatic training")
        return

    # start background thread to train and load
    t = threading.Thread(target=_background_train_and_load, daemon=True)
    t.start()


def _build_feature_vector(input_dict: dict, feature_names: list) -> pd.DataFrame:
    """Construct a single-row DataFrame with columns in feature_names using values from
    input_dict where possible. Missing features are filled with sensible defaults (0).

    This is a best-effort mapping: features that require global aggregates (customer history,
    ip counts) are set to 0 when not provided.
    """
    # start with zeros
    row = {f: 0.0 for f in feature_names}

    # direct mappings
    # transaction amount
    if "amount" in input_dict:
        amt = float(input_dict.get("amount", 0.0) or 0.0)
        if "transaction_amount" in feature_names:
            row["transaction_amount"] = amt
        if "log_amount" in feature_names:
            row["log_amount"] = math.log1p(max(0.0, amt))

    # account age
    if "account_age" in input_dict:
        acc = float(input_dict.get("account_age", 0) or 0)
        if "account_age_days" in feature_names:
            row["account_age_days"] = acc
        # account_age_bucket approximate using same bins as training
        if "account_age_bucket" in feature_names:
            bins = [-1, 30, 90, 180, 365, 730, 20000]
            try:
                row["account_age_bucket"] = int(pd.cut([acc], bins=bins).codes[0])
            except Exception:
                row["account_age_bucket"] = 0

    # customer age
    if "age" in input_dict:
        age = float(input_dict.get("age", 0) or 0)
        if "customer_age_bucket" in feature_names:
            bins = [0, 20, 30, 40, 50, 60, 120]
            try:
                row["customer_age_bucket"] = int(pd.cut([age], bins=bins, include_lowest=True).codes[0])
            except Exception:
                row["customer_age_bucket"] = 0

        # set raw customer_age if model expects it
        if "customer_age" in feature_names:
            row["customer_age"] = age

    # transaction hour
    if "transaction_hour" in input_dict:
        hr = int(input_dict.get("transaction_hour", 0) or 0)
        if "hour_sin" in feature_names:
            row["hour_sin"] = math.sin(2 * math.pi * hr / 24)
        if "hour_cos" in feature_names:
            row["hour_cos"] = math.cos(2 * math.pi * hr / 24)
        if "is_night" in feature_names:
            row["is_night"] = 1 if (hr <= 6 or hr >= 22) else 0
        # also set raw transaction_hour if model expects it
        if "transaction_hour" in feature_names:
            row["transaction_hour"] = hr

    # device / payment method / ip exist as categorical-derived stats in training; set to 0
    # but allow user to provide simple proxies
    if "cust_total_txn" in input_dict and "cust_total_txn" in feature_names:
        row["cust_total_txn"] = float(input_dict.get("cust_total_txn", 0) or 0)
        if "cust_total_txn_log" in feature_names:
            row["cust_total_txn_log"] = math.log1p(max(0.0, row["cust_total_txn"]))

    # amount_per_unit fallback
    if "amount_per_unit" in feature_names and "amount_per_unit" in input_dict:
        row["amount_per_unit"] = float(input_dict.get("amount_per_unit", 0) or 0)

    # quantity / units
    if ("quantity" in feature_names or "qty" in feature_names) and ("quantity" in input_dict or "qty" in input_dict):
        q = int(input_dict.get("quantity", input_dict.get("qty", 0)) or 0)
        if "quantity" in feature_names:
            row["quantity"] = q

    # final: ensure numeric dtype
    df = pd.DataFrame([row], columns=feature_names)
    df = df.fillna(0.0)
    return df


def _normalize_name(s: str) -> str:
    """Normalize human-readable feature names to snake_case-like keys.

    Example: 'Transaction Amount' -> 'transaction_amount'
    """
    if not isinstance(s, str):
        return s
    s2 = s.strip().lower()
    # replace non-alphanumeric with underscore
    s2 = re.sub(r"[^0-9a-z]+", "_", s2)
    # collapse multiple underscores
    s2 = re.sub(r"_+", "_", s2)
    # strip leading/trailing underscores
    s2 = s2.strip("_")
    return s2

class UserData(BaseModel):
    age: int
    account_age: int
    total_transactions: int
    past_fraud: int
    avg_order_value: float

@app.post("/predict_user")
def predict_user(data: UserData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

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


@app.get("/model_status")
def model_status():
    """Return whether a model is loaded and any metadata available.

    This endpoint is intentionally lightweight and does not block on training.
    """
    loaded = model is not None
    meta = metadata
    # if metadata not present in-memory, try to read metadata.json as a fallback
    if meta is None:
        try:
            meta_path = Path("models") / "metadata.json"
            if meta_path.exists() and meta_path.stat().st_size > 0:
                import json as _json

                with open(meta_path, "r") as f:
                    meta = _json.load(f)
        except Exception:
            meta = None

    message = None if loaded else "Model not loaded"

    # Add lightweight diagnostics about the loaded model to help debugging
    diagnostics = {}
    try:
        if loaded and model is not None:
            diagnostics["model_type"] = type(model).__name__
            # if our EnsemblePredictor wrapper is present, inspect inner estimators
            try:
                est = getattr(model, "lgbm", None) or getattr(model, "xgb", None) or model
                diagnostics["estimator_type"] = type(est).__name__
                if hasattr(est, "n_features_in_"):
                    diagnostics["n_features_in_"] = int(getattr(est, "n_features_in_"))
                if hasattr(est, "feature_names_in_"):
                    diagnostics["feature_names_in_"] = list(getattr(est, "feature_names_in_"))
                if hasattr(est, "feature_name_"):
                    diagnostics["feature_name_"] = list(getattr(est, "feature_name_"))
            except Exception:
                diagnostics["estimator_inspect_error"] = True
    except Exception:
        diagnostics["diagnostics_error"] = True

    return {"loaded": bool(loaded), "metadata": meta, "message": message, "diagnostics": diagnostics}


@app.post("/promote_model")
def promote_model(x_admin_token: str | None = Header(default=None)):
    """Promote models/fraud_model.pkl.bak -> models/fraud_model.pkl and reload model.

    If ADMIN_TOKEN is set in the environment, require the same value in the
    X-Admin-Token header. This endpoint is intended for admin/repair usage.
    """
    admin_token = os.getenv("ADMIN_TOKEN")
    if admin_token:
        if x_admin_token != admin_token:
            raise HTTPException(status_code=403, detail="Missing or invalid admin token")

    bak = Path("models") / "fraud_model.pkl.bak"
    dest = Path("models") / "fraud_model.pkl"
    if not bak.exists():
        return {"status": "error", "message": f"Backup file not found: {bak}"}

    try:
        os.makedirs(dest.parent, exist_ok=True)
        # move atomically when possible
        try:
            os.replace(str(bak), str(dest))
        except Exception:
            shutil.copyfile(str(bak), str(dest))
            os.remove(str(bak))

        # reload into memory using the loader
        global model, encoder, metadata
        try:
            model, encoder, metadata = load_model()
        except Exception:
            # if reloading fails, leave on-disk file and return error
            tb = traceback.format_exc()
            return {"status": "error", "message": "Failed to reload model after promotion", "traceback": tb}

        return {"status": "ok", "saved_to": str(dest)}
    except Exception as e:
        tb = traceback.format_exc()
        return {"status": "error", "message": str(e), "traceback": tb}


@app.post("/train")
def train_model():
    """Trigger training by running the existing bt4012_model training script.

    This dynamically imports and runs the training routine, then saves an ensemble
    `models/fraud_model.pkl` that the API endpoints use. The existing bt4012_model
    file is not modified.
    """
    global model
    # trigger training synchronously for the /train endpoint and update training_progress
    try:
        training_progress["in_progress"] = True
        training_progress["percent"] = 0
        training_progress["message"] = "starting training"
        training_progress["start_time"] = time.time()
        bundle = _load_and_run_bt4012_model()
        training_progress["percent"] = 95
        training_progress["message"] = "finalizing and saving model"
    except Exception as e:
        tb = traceback.format_exc()
        training_progress["in_progress"] = False
        training_progress["message"] = f"error: {e}"
        return {"status": "error", "error": str(e), "traceback": tb}

    # Prefer a pre-built model object from the training bundle if present
    model_obj = bundle.get("model")
    if model_obj is None:
        # If not present, try to construct an ensemble from returned lgbm/xgb
        lgbm = bundle.get("lgbm")
        xgb = bundle.get("xgb")
        if lgbm is None or xgb is None:
            return {"status": "error", "error": "Training finished but returned no models."}

        # if the bundle provides ensemble_weights, prefer them
        ew = bundle.get("ensemble_weights") or {"lgbm": 0.6, "xgb": 0.4}
        w_lgbm = float(ew.get("lgbm", 0.6))
        w_xgb = float(ew.get("xgb", 0.4))

        model_obj = EnsemblePredictor(lgbm, xgb, w_lgbm=w_lgbm, w_xgb=w_xgb)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fraud_model.pkl"
    joblib.dump(model_obj, model_path)

    # Save metadata (feature names, ensemble weights, threshold) for later inference
    try:
        meta = {
            "feature_names": bundle.get("feature_names", []),
            "ensemble_weights": bundle.get("ensemble_weights", {}),
            "best_threshold": bundle.get("best_threshold"),
        }
        with open(model_dir / "metadata.json", "w") as mf:
            json.dump(meta, mf)
    except Exception:
        print("Warning: failed to write metadata.json")

    # update in-memory model used by endpoints
    model = model_obj

    # optional metrics to return
    metrics = bundle.get("valid_metrics") or bundle.get("test_metrics") or {}
    # mark training complete
    training_progress["in_progress"] = False
    training_progress["percent"] = 100
    training_progress["message"] = "completed"
    return {"status": "ok", "saved_to": str(model_path), "metrics": metrics}


@app.get("/train_status")
def train_status():
    """Return the current training progress information for the frontend to poll.

    This is a lightweight endpoint that returns a small JSON describing whether
    training is in progress, an approximate percent (0-100) and an optional
    message. The percent is best-effort and may be coarse unless the trainer
    provides fine-grained updates via a progress callback.
    """
    return training_progress


@app.post("/predict_fraud")
def predict_fraud(payload: dict):
    """Predict fraud probability using the loaded fraud_model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train or promote a model first.")

    # debug log incoming payload keys and metadata
    try:
        logger.info("predict_fraud called; payload keys=%s; metadata_keys=%s", list(payload.keys()) if isinstance(payload, dict) else None, list(metadata.keys()) if isinstance(metadata, dict) else None)
    except Exception:
        pass

    # Helper: try to obtain canonical feature order (may be human-readable names)
    fnames = None
    orig_fnames = None
    if metadata and isinstance(metadata, dict):
        orig_fnames = metadata.get("feature_names")
        if orig_fnames:
            # create normalized internal names from human-readable metadata
            fnames = [_normalize_name(x) for x in orig_fnames]

    # If client provided explicit features, prefer that (no remapping)
    if "features" in payload and isinstance(payload["features"], dict):
        features = payload["features"]
        if fnames and orig_fnames:
            # build a normalized row using orig_fnames -> normalized fnames mapping
            row = {}
            for orig, norm in zip(orig_fnames, fnames):
                # prefer user-provided key (orig readable) then try normalized key
                val = None
                if orig in features:
                    val = features.get(orig)
                elif norm in features:
                    val = features.get(norm)
                row[norm] = float(val or 0.0)
            df = pd.DataFrame([row], columns=fnames).fillna(0.0)
        elif fnames:
            # no orig mapping, but we have normalized names list
            row = {c: float(payload["features"].get(c, 0.0) or 0.0) for c in fnames}
            df = pd.DataFrame([row], columns=fnames).fillna(0.0)
        else:
            # use keys as provided
            cols = list(features.keys())
            df = pd.DataFrame([features], columns=cols).fillna(0.0)

    else:
        # Expect either 'user' or 'transaction' mapping
        if "transaction" in payload:
            input_map = payload.get("transaction", {})
        elif "user" in payload:
            input_map = payload.get("user", {})
        else:
            raise HTTPException(status_code=400, detail="Payload must include either 'features', 'user' or 'transaction' key")

        if not fnames:
            # try to infer feature names from underlying estimators if metadata is absent
            try:
                est = getattr(model, "lgbm", None) or getattr(model, "xgb", None) or model
                if hasattr(est, "feature_name_"):
                    fnames = list(getattr(est, "feature_name_"))
                elif hasattr(est, "feature_names_in_"):
                    fnames = list(getattr(est, "feature_names_in_"))
            except Exception:
                fnames = None

        if not fnames:
            # we cannot build a mapped vector without feature names
            raise HTTPException(status_code=400, detail="Model feature names unknown; provide full 'features' dict in request")

        # Build using the canonical (normalized) feature list
        df = _build_feature_vector(input_map, fnames)

    # Align dataframe columns to what the model expects (by name when possible,
    # otherwise by n_features_in_) before calling predict_proba. This lets us
    # perform predictions without re-running training.
    try:
        # pick an underlying estimator to inspect (ensemble wrappers expose lgbm/xgb)
        est = getattr(model, "lgbm", None) or getattr(model, "xgb", None) or model

        # preferred: explicit feature name list from estimator or metadata
        expected_names = None
        try:
            if hasattr(est, "feature_names_in_"):
                expected_names = list(getattr(est, "feature_names_in_"))
            elif hasattr(est, "feature_name_"):
                expected_names = list(getattr(est, "feature_name_"))
        except Exception:
            expected_names = None

        if not expected_names and metadata and isinstance(metadata, dict):
            expected_names = metadata.get("feature_names")

        if expected_names:
            norm_expected = [_normalize_name(x) for x in expected_names]
            df_out = pd.DataFrame()
            for exp_name, exp_norm in zip(expected_names, norm_expected):
                if exp_norm in df.columns:
                    df_out[exp_name] = df[exp_norm]
                else:
                    # missing -> zero
                    df_out[exp_name] = 0.0
            df = df_out
        else:
            # fallback: use n_features_in_ if available to pad/trim columns
            n_in = getattr(est, "n_features_in_", None)
            if n_in is not None:
                cur = df.shape[1]
                if cur < n_in:
                    for i in range(n_in - cur):
                        df[f"_pad_{i}"] = 0.0
                elif cur > n_in:
                    df = df.iloc[:, :n_in]
            else:
                raise HTTPException(status_code=400, detail="Cannot determine model input shape; provide full 'features' dict in request")

        # optional debug: return the built feature vector back to caller
        if isinstance(payload, dict) and payload.get("debug"):
            return {"built_features": {"columns": list(df.columns), "values": df.values.tolist()}}

        X = df.values if hasattr(df, "values") else df
        prob = float(model.predict_proba(X)[0][1]) * 100.0
        # include built feature vector for debugging/inspection
        return {"fraud_probability": round(prob, 2), "built_features": {"columns": list(df.columns), "values": df.values.tolist()}}
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}\n{tb}")
