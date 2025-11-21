# backend/model_loader.py
import joblib
import os
import traceback

MODEL_PATH = "models/fraud_model.pkl"
MODEL_BAK_PATH = "models/fraud_model.pkl.bak"
ENCODER_PATH = "models/encoder.pkl"


def load_model():
    """Attempt to load persisted model and encoder.

    If the model file is missing or corrupted (EOFError / UnpicklingError),
    return (None, None) and log the traceback. The calling code should handle
    the case where model is None and allow training to proceed.
    """
    model = None
    encoder = None

    try:
        # Prefer the canonical .pkl file. If missing, fallback to .pkl.bak if present.
        chosen = None
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            chosen = MODEL_PATH
        elif os.path.exists(MODEL_BAK_PATH) and os.path.getsize(MODEL_BAK_PATH) > 0:
            chosen = MODEL_BAK_PATH

        if chosen is None:
            print(f"Model file {MODEL_PATH} not found. Starting without a model.")
        else:
            try:
                model = joblib.load(chosen)
                if chosen != MODEL_PATH:
                    print(f"Loaded model from fallback file {chosen}")
                    # attempt to promote the .bak to the canonical .pkl so subsequent
                    # runs and other tooling use the expected filename.
                    try:
                        import shutil

                        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                        shutil.copyfile(chosen, MODEL_PATH)
                        print(f"Promoted {chosen} -> {MODEL_PATH}")
                    except Exception:
                        print(f"Warning: failed to copy {chosen} to {MODEL_PATH}")
                        traceback.print_exc()
            except Exception:
                # The file may be corrupted/truncated (EOFError on unpickle). Move it aside
                # so subsequent restarts don't repeatedly attempt to load the same bad file.
                print(f"Failed to load model from {chosen} — attempting to continue without model.")
                traceback.print_exc()
                try:
                    corrupted_path = chosen + ".corrupt"
                    print(f"Renaming corrupted model file {chosen} -> {corrupted_path}")
                    os.replace(chosen, corrupted_path)
                except Exception:
                    print(f"Warning: unable to rename corrupted file {chosen}")
                    traceback.print_exc()
    except Exception:
        print(f"Failed to load model from {MODEL_PATH}:")
        traceback.print_exc()
        model = None

    try:
        if os.path.exists(ENCODER_PATH) and os.path.getsize(ENCODER_PATH) > 0:
            encoder = joblib.load(ENCODER_PATH)
    except Exception:
        print(f"Failed to load encoder from {ENCODER_PATH}:")
        traceback.print_exc()
        encoder = None

    # Try to load metadata if present
    metadata = None
    try:
        meta_path = "models/metadata.json"
        if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
            import json

            with open(meta_path, "r") as f:
                metadata = json.load(f)
    except Exception:
        print("Failed to load metadata.json:")
        traceback.print_exc()

    # If we loaded a model and metadata is present, try to make the metadata
    # align with the estimator's expected input names. Prefer estimator-provided
    # feature names (feature_names_in_ / feature_name_) when available. If not
    # available, perform a conservative sanitization of the on-disk metadata to
    # remove obvious label/target columns (e.g. 'is fraud', 'label', etc.).
    try:
        if model is not None and metadata is not None and isinstance(metadata, dict):
            est = getattr(model, "lgbm", None) or getattr(model, "xgb", None) or model
            inferred = None
            try:
                if hasattr(est, "feature_names_in_"):
                    inferred = list(getattr(est, "feature_names_in_"))
                elif hasattr(est, "feature_name_"):
                    inferred = list(getattr(est, "feature_name_"))
            except Exception:
                inferred = None

            if inferred:
                # Overwrite metadata feature_names with the estimator's canonical names
                metadata["feature_names"] = inferred
            else:
                # conservative sanitization: drop feature names that look like a target
                fn = metadata.get("feature_names")
                if isinstance(fn, list):
                    cleaned = []
                    for name in fn:
                        if not isinstance(name, str):
                            continue
                        low = name.strip().lower()
                        if any(tok in low for tok in ("fraud", "is fraudulent", "is_fraud", "label", "target")):
                            # skip obvious target-like column
                            continue
                        cleaned.append(name)
                    metadata["feature_names"] = cleaned
    except Exception:
        # non-fatal: keep metadata as originally loaded
        traceback.print_exc()

    return model, encoder, metadata
