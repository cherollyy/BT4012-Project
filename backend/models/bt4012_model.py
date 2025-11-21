
import os, gc, math, random
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

current_file_path = Path(__file__).resolve()
ROOT = current_file_path.parent.parent.parent

# Use /app/data in container, fallback to local data folder for development
PREP_FILE = "Fraudulent_E-Commerce_Transaction_Data_MERGED.csv"
CONTAINER_DATA_PATH = Path("/app/data", PREP_FILE)
LOCAL_DATA_PATH = Path(ROOT, "data", PREP_FILE)
PREP_PATH = CONTAINER_DATA_PATH if CONTAINER_DATA_PATH.exists() else LOCAL_DATA_PATH

def print_head(title, char="=", width=80):
    line = char * width
    print(f"\n{line}\n{title}\n{line}")


def add_target_encoding_kfold(X_train, y_train, X_valid, X_test, col, n_splits=5, smoothing=100):
    print(f"  - Target encoding: {col}")
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()

    global_mean = y_train.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    col_tr_te = np.zeros(len(X_train), dtype=float)

    for tr_idx, hold_idx in kf.split(X_train):
        tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]

        stats = tr.groupby(col)[y_tr.name].agg(["mean", "count"])
        means = stats["mean"]
        counts = stats["count"]
        smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
        col_tr_te[hold_idx] = (
            X_train.iloc[hold_idx][col].map(smooth).fillna(global_mean)
        )

    X_train[col + "_te"] = col_tr_te

    stats_full = X_train.groupby(col)[col + "_te"].mean()
    X_valid[col + "_te"] = X_valid[col].map(stats_full).fillna(global_mean)
    X_test[col + "_te"] = X_test[col].map(stats_full).fillna(global_mean)

    return X_train, X_valid, X_test, col + "_te"


def find_best_threshold(y_true, prob, start=0.1, end=0.9, step=0.01):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(start, end + 1e-9, step):
        pred = (prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t), float(best_f1)


def evaluate_all(y_true, prob, threshold):
    auc = roc_auc_score(y_true, prob)
    pred = (prob >= threshold).astype(int)
    acc = accuracy_score(y_true, pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def load_preprocessed_main():
    print_head("1) Load preprocessed fraud dataset")
    df = pd.read_csv(PREP_PATH)

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        print(f"  Dropping columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    print("[Preprocessed] Shape:", df.shape)
    return df


def run_high_perf_tree_ensemble_model_only(random_state=42, progress_callback=None):
    df = load_preprocessed_main()
    # notify progress if callback provided
    try:
        if progress_callback:
            progress_callback(10, "loaded data")
    except Exception:
        pass
    
    # Determine target column. Prefer 'is_fraudulent', but fall back to any
    # column that contains the token 'fraud' (case-insensitive), e.g. 'Is Fraudulent'.
    target_col = "is_fraudulent"
    if target_col not in df.columns:
        # try to find a fraud-like column name
        cand = next((c for c in df.columns if "fraud" in c.lower()), None)
        if cand:
            target_col = cand
            print(f"Using detected target column: {target_col}")
        else:
            raise ValueError(f"Target column 'is_fraudulent' not found in preprocessed CSV.")

    # Load target and coerce to binary 0/1 if necessary (handles 'Yes'/'No', True/False, etc.)
    y = df[target_col]
    try:
        if y.dtype == object:
            ys = y.astype(str).str.strip().str.lower()
            mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0, "y": 1, "n": 0}
            y = ys.map(mapping).fillna(pd.to_numeric(ys, errors="coerce")).fillna(0)
        else:
            y = pd.to_numeric(y, errors="coerce").fillna(0)
    except Exception:
        y = pd.to_numeric(y, errors="coerce").fillna(0)

    y = y.astype(int)

    print_head("2) Prepare train/valid/test split")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=random_state
    )
    tr_idx, test_idx = next(splitter.split(df, y))

    df_trv = df.iloc[tr_idx].reset_index(drop=True)
    y_trv = y.iloc[tr_idx].reset_index(drop=True)

    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=random_state
    )
    tr_idx2, valid_idx = next(splitter2.split(df_trv, y_trv))

    train = df_trv.iloc[tr_idx2].reset_index(drop=True)
    valid = df_trv.iloc[valid_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    y_train = y_trv.iloc[tr_idx2].reset_index(drop=True)
    y_valid = y_trv.iloc[valid_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    print(f"[Split] Train: {train.shape}, Valid: {valid.shape}, Test: {test.shape}")
    try:
        if progress_callback:
            progress_callback(30, "prepared splits")
    except Exception:
        pass
    
    print_head("3) Encoding (Frequency + Target)")

    cat_cols = []
    for c in [
        "customer_id",
        "ip_address",
        "device_used",
        "customer_location",
        "payment_method",
        "product_category",
    ]:
        if c in train.columns:
            cat_cols.append(c)

    for c in cat_cols:
        freq = train[c].value_counts()
        train[c + "_freq"] = train[c].map(freq).astype(float)
        valid[c + "_freq"] = valid[c].map(freq).fillna(0).astype(float)
        test[c + "_freq"] = test[c].map(freq).fillna(0).astype(float)

    for c in cat_cols:
        train, valid, test, _ = add_target_encoding_kfold(
            train, y_train, valid, test, c, n_splits=5, smoothing=100
        )

    print_head("4) Build feature matrix")

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    print(f"[Features] numeric feature count: {len(numeric_cols)}")
    try:
        if progress_callback:
            progress_callback(45, "built feature matrix")
    except Exception:
        pass
    
    X_train = train[numeric_cols].astype(float)
    X_valid = valid[numeric_cols].astype(float)
    X_test = test[numeric_cols].astype(float)

    for X in [X_train, X_valid, X_test]:
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0.0, inplace=True)

    pos_rate = y_train.mean()
    neg_pos_ratio = (1 - pos_rate) / max(pos_rate, 1e-6)
    print(f"pos_rate: {pos_rate:.4f}, scale_pos_weight ~ {neg_pos_ratio:.2f}")

    print_head("5) Train LightGBM + XGBoost (Ensemble)")

    lgbm = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=3.0,
        objective="binary",
        class_weight=None,
        n_jobs=-1,
        random_state=random_state,
    )

    # If provided, attach a simple LightGBM callback to report training progress
    lgb_callbacks = []
    try:
        if progress_callback:
            total = int(getattr(lgbm, "n_estimators", 1200) or 1200)

            def _lgb_progress(env):
                try:
                    itr = getattr(env, "iteration", None)
                    if itr is None:
                        itr = 0
                    # map iteration to percent range 50-75
                    pct = 50 + int(25 * min(1.0, (itr + 1) / max(1, total)))
                    progress_callback(pct, f"lgbm iter {itr+1}/{total}")
                except Exception:
                    pass

            lgb_callbacks = [ _lgb_progress ]
    except Exception:
        lgb_callbacks = []

    progress_callback and progress_callback(50, "starting LGBM fit")
    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=lgb_callbacks,
    )
    try:
        if progress_callback:
            progress_callback(75, "lgbm finished")
    except Exception:
        pass

    xgb = XGBClassifier(
        n_estimators=900,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_alpha=0.5,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=random_state,
        tree_method="hist",
        scale_pos_weight=neg_pos_ratio,
    )

    try:
        if progress_callback:
            progress_callback(80, "starting xgb fit")
    except Exception:
        pass

    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    try:
        if progress_callback:
            progress_callback(92, "xgb finished")
    except Exception:
        pass

    print_head("6) Predict & threshold tuning")

    valid_prob = 0.6 * lgbm.predict_proba(X_valid)[:, 1] + \
                 0.4 * xgb.predict_proba(X_valid)[:, 1]
    test_prob = 0.6 * lgbm.predict_proba(X_test)[:, 1] + \
                0.4 * xgb.predict_proba(X_test)[:, 1]

    best_t, best_f1 = find_best_threshold(
        y_valid, valid_prob, start=0.1, end=0.9, step=0.01
    )
    try:
        if progress_callback:
            progress_callback(95, "threshold tuning")
    except Exception:
        pass
    print(f"Best threshold on VALID: {best_t:.4f}, Best F1: {best_f1:.4f}")

    print_head(f"7) Final Evaluation (threshold={best_t:.4f})")
    valid_metrics = evaluate_all(y_valid, valid_prob, best_t)
    test_metrics = evaluate_all(y_test, test_prob, best_t)

    print("==== VALID Metrics ====")
    for k, v in valid_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    print("\n==== TEST Metrics ====")
    for k, v in test_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    try:
        if progress_callback:
            progress_callback(98, "finalizing")
    except Exception:
        pass

    # Return the trained base estimators and metadata. Do NOT return
    # a locally-defined EnsemblePredictor instance (nested classes are
    # not picklable). The API caller (`backend/main.py`) will construct
    # a top-level EnsemblePredictor for serialization if needed.
    return {
        "lgbm": lgbm,
        "xgb": xgb,
        "best_threshold": best_t,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "feature_names": numeric_cols,
    }




if __name__ == "__main__":
    # Allow the module to be executed directly for debugging/training, but
    # avoid running training on import (imports should be side-effect free).
    model_bundle = run_high_perf_tree_ensemble_model_only(random_state=42)
    gc.collect()
