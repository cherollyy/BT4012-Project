#!/usr/bin/env python3
"""Generate models/metadata.json by inspecting the preprocessed CSV used by bt4012_model.

This avoids retraining: it extracts numeric feature column names the model expects
based on the training preprocessing logic and writes a minimal metadata.json that
lets the API map short payloads into a feature vector.

Run from repo root or inside container:

# host
python backend/scripts/generate_metadata.py

# inside docker
docker compose run --rm backend python backend/scripts/generate_metadata.py

"""
import json
from pathlib import Path
import sys

# attempt to import the training module
try:
    from backend.models import bt4012_model as m
except Exception:
    # try alternative import if running from repo root
    try:
        import importlib.util
        import sys as _sys
        p = Path(__file__).resolve().parents[1] / "models" / "bt4012_model.py"
        spec = importlib.util.spec_from_file_location("bt4012_model", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m = mod
    except Exception as e:
        print("Failed to import bt4012_model:", e)
        raise

print("Loading preprocessed dataset to infer feature names (may be large)...")
try:
    df = m.load_preprocessed_main()
except Exception as e:
    print("Failed to load preprocessed CSV via bt4012_model.load_preprocessed_main():", e)
    raise

# same logic as in bt4012_model: numeric columns excluding the target
target_col = "is_fraudulent"
import numpy as np
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

meta = {
    "feature_names": numeric_cols,
    "ensemble_weights": {"lgbm": 0.6, "xgb": 0.4},
    "best_threshold": None,
}

out = Path("backend") / "models" / "metadata.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Wrote metadata.json with {len(numeric_cols)} feature names to: {out}")
print("Sample features:", numeric_cols[:20])
print("Now restart the backend or call the promote endpoint to reload metadata.")
