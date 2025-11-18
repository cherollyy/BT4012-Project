
import os, gc, math, random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

ROOT = "/content/drive/MyDrive/BT4012"




def print_head(title, char="=", width=80):
    line = char * width
    print(f"\n{line}\n{title}\n{line}")

def safe_lower_strip(series):
    return series.astype(str).str.lower().str.strip()

def safe_int_from_codes(series, fill_value=0):
    """
    NaN, inf를 전부 처리한 뒤 int로 변환.
    (주로 bucket / codes용)
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.fillna(fill_value)
    return s.astype(int)



def build_counterfeit_risk_features():
    print_head("1) Build counterfeit risk features from sub-datasets")

    path_txn = os.path.join(ROOT, "_counterfeit_transactions.csv")
    path_prod = os.path.join(ROOT, "counterfeit_products.csv")

    df_txn = pd.read_csv(path_txn)
    df_prod = pd.read_csv(path_prod)

    df_txn["payment_method_norm"] = safe_lower_strip(df_txn["payment_method"])
    if "customer_location" in df_txn.columns:
        df_txn["customer_location_norm"] = safe_lower_strip(df_txn["customer_location"])
    if "shipping_speed" in df_txn.columns:
        df_txn["shipping_speed_norm"] = safe_lower_strip(df_txn["shipping_speed"])


    stats_txn_pm = (
        df_txn.groupby("payment_method_norm")["involves_counterfeit"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "cnt_txn_counterfeit_rate_pm",
                "count": "cnt_txn_count_pm",
            }
        )
    )

    if "customer_location_norm" in df_txn.columns:
        stats_txn_loc_pm = (
            df_txn.groupby(["customer_location_norm", "payment_method_norm"])[
                "involves_counterfeit"
            ]
            .agg(["mean", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "cnt_txn_counterfeit_rate_loc_pm",
                    "count": "cnt_txn_count_loc_pm",
                }
            )
        )
    else:
        stats_txn_loc_pm = None

    print("\n[Counterfeit Txn stats by payment_method]")
    print(stats_txn_pm.head())

    df_prod["category_norm"] = safe_lower_strip(df_prod["category"])
    if "shipping_origin" in df_prod.columns:
        df_prod["shipping_origin_norm"] = safe_lower_strip(df_prod["shipping_origin"])

    stats_prod_cat = (
        df_prod.groupby("category_norm")["is_counterfeit"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "cnt_prod_counterfeit_rate_cat",
                "count": "cnt_prod_count_cat",
            }
        )
    )

    if "shipping_origin_norm" in df_prod.columns:
        stats_prod_cat_org = (
            df_prod.groupby(["category_norm", "shipping_origin_norm"])[
                "is_counterfeit"
            ]
            .agg(["mean", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "cnt_prod_counterfeit_rate_cat_org",
                    "count": "cnt_prod_count_cat_org",
                }
            )
        )
    else:
        stats_prod_cat_org = None

    print("\n[Counterfeit Prod stats by category]")
    print(stats_prod_cat.head())

    return {
        "stats_txn_pm": stats_txn_pm,
        "stats_txn_loc_pm": stats_txn_loc_pm,
        "stats_prod_cat": stats_prod_cat,
        "stats_prod_cat_org": stats_prod_cat_org,
    }


def load_and_feature_main(counter_stats):
    print_head("2) Load & enrich main fraud dataset")

    path_main = os.path.join(ROOT, "Fraudulent_E-Commerce_Transaction_Data_FULL.csv")
    df = pd.read_csv(path_main)

    df = df.rename(columns=lambda c: c.strip().replace(" ", "_").replace("-", "_"))

    rename_map = {}
    if "Transaction_Amount" in df.columns:
        rename_map["Transaction_Amount"] = "transaction_amount"
    if "Transaction_Date" in df.columns:
        rename_map["Transaction_Date"] = "transaction_date"
    if "Customer_ID" in df.columns:
        rename_map["Customer_ID"] = "customer_id"
    if "Customer_Age" in df.columns:
        rename_map["Customer_Age"] = "customer_age"
    if "Customer_Location" in df.columns:
        rename_map["Customer_Location"] = "customer_location"
    if "Payment_Method" in df.columns:
        rename_map["Payment_Method"] = "payment_method"
    if "Product_Category" in df.columns:
        rename_map["Product_Category"] = "product_category"
    if "Device_Used" in df.columns:
        rename_map["Device_Used"] = "device_used"
    if "IP_Address" in df.columns:
        rename_map["IP_Address"] = "ip_address"
    if "Shipping_Address" in df.columns:
        rename_map["Shipping_Address"] = "shipping_address"
    if "Billing_Address" in df.columns:
        rename_map["Billing_Address"] = "billing_address"
    if "Account_Age_Days" in df.columns:
        rename_map["Account_Age_Days"] = "account_age_days"
    if "Transaction_Hour" in df.columns:
        rename_map["Transaction_Hour"] = "transaction_hour"
    if "Is_Fraudulent" in df.columns:
        rename_map["Is_Fraudulent"] = "is_fraudulent"

    df = df.rename(columns=rename_map)

    for col in ["payment_method", "product_category", "customer_location"]:
        if col in df.columns:
            df[col + "_norm"] = safe_lower_strip(df[col])

    s = counter_stats

    if "payment_method_norm" in df.columns:
        df = df.merge(
            s["stats_txn_pm"],
            how="left",
            on="payment_method_norm",
        )

    if s["stats_txn_loc_pm"] is not None and "customer_location_norm" in df.columns:
        df = df.merge(
            s["stats_txn_loc_pm"],
            how="left",
            on=["customer_location_norm", "payment_method_norm"],
        )

    if "product_category_norm" in df.columns:
        left_key = "product_category_norm"
    elif "product_category" in df.columns:
        df["product_category_norm"] = safe_lower_strip(df["product_category"])
        left_key = "product_category_norm"
    else:
        left_key = None

    if left_key is not None:
        df = df.merge(
            s["stats_prod_cat"],
            how="left",
            left_on=left_key,
            right_on="category_norm",
        )

    for col in [
        "cnt_txn_counterfeit_rate_pm",
        "cnt_txn_count_pm",
        "cnt_txn_counterfeit_rate_loc_pm",
        "cnt_txn_count_loc_pm",
        "cnt_prod_counterfeit_rate_cat",
        "cnt_prod_count_cat",
        "cnt_prod_counterfeit_rate_cat_org",
        "cnt_prod_count_cat_org",
    ]:
        if col in df.columns:
            if "rate" in col:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(0)

    print("[Main] Shape after merge:", df.shape)


    print("\nCreating fraud-style engineered features...")


    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["tx_date"] = df["transaction_date"].dt.date
        df["tx_dayofweek"] = df["transaction_date"].dt.weekday
        df["tx_day"] = df["transaction_date"].dt.day
        df["tx_month"] = df["transaction_date"].dt.month
    else:
        df["tx_date"] = pd.NaT
        df["tx_dayofweek"] = -1
        df["tx_day"] = -1
        df["tx_month"] = -1

    if "transaction_hour" in df.columns:
        df["transaction_hour"] = pd.to_numeric(df["transaction_hour"], errors="coerce")
        df["transaction_hour"] = df["transaction_hour"].fillna(-1).astype(int)
        hour = df["transaction_hour"].clip(0, 23)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["is_night"] = ((hour <= 6) | (hour >= 22)).astype(int)
    else:
        df["transaction_hour"] = -1
        df["hour_sin"] = 0
        df["hour_cos"] = 0
        df["is_night"] = 0

    df["is_weekend"] = df.get("tx_dayofweek", -1).isin([5, 6]).astype(int)

    if "transaction_amount" in df.columns:
        df["transaction_amount"] = df["transaction_amount"].fillna(0.0)
        df["log_amount"] = np.log1p(df["transaction_amount"].clip(lower=0))
        high_th = df["transaction_amount"].quantile(0.95)
        df["is_high_amount"] = (df["transaction_amount"] >= high_th).astype(int)
    else:
        df["transaction_amount"] = 0.0
        df["log_amount"] = 0.0
        df["is_high_amount"] = 0

    if "Quantity" in df.columns and "transaction_amount" in df.columns:
        df["quantity"] = df["Quantity"]
    if "quantity" in df.columns:
        df["amount_per_unit"] = df["transaction_amount"] / df["quantity"].replace(0, np.nan)
        df["amount_per_unit"] = df["amount_per_unit"].replace([np.inf, -np.inf], np.nan)
        df["amount_per_unit"] = df["amount_per_unit"].fillna(df["transaction_amount"])
    else:
        df["amount_per_unit"] = df["transaction_amount"]

    if "account_age_days" in df.columns:
        df["account_age_days"] = pd.to_numeric(df["account_age_days"], errors="coerce")
        df["account_age_days"] = df["account_age_days"].fillna(
            df["account_age_days"].median()
        )
        cats = pd.cut(
            df["account_age_days"],
            bins=[-1, 30, 90, 180, 365, 730, 20000],
        )
        df["account_age_bucket"] = cats.cat.codes  
    else:
        df["account_age_days"] = 0.0
        df["account_age_bucket"] = 0

    if "customer_age" in df.columns:
        df["customer_age"] = pd.to_numeric(df["customer_age"], errors="coerce")
        df["customer_age"] = df["customer_age"].fillna(df["customer_age"].median())
        cats = pd.cut(
            df["customer_age"],
            bins=[0, 20, 30, 40, 50, 60, 120],
            include_lowest=True,
        )
        df["customer_age_bucket"] = cats.cat.codes
    else:
        df["customer_age_bucket"] = 0

    if "customer_id" in df.columns and "tx_date" in df.columns:
        grp = df.groupby(["customer_id", "tx_date"])["transaction_amount"].transform("count")
        df["cust_txn_per_day"] = grp.astype(float)
        df["cust_txn_per_day_log"] = np.log1p(df["cust_txn_per_day"])
    else:
        df["cust_txn_per_day_log"] = 0.0

    if "customer_id" in df.columns:
        cust_cnt = df.groupby("customer_id")["transaction_amount"].transform("count")
        cust_amt_sum = df.groupby("customer_id")["transaction_amount"].transform("sum")
        df["cust_total_txn"] = cust_cnt.astype(float)
        df["cust_total_amt"] = cust_amt_sum.astype(float)
        df["cust_avg_amt"] = df["cust_total_amt"] / df["cust_total_txn"].replace(0, np.nan)
        df["cust_avg_amt"] = df["cust_avg_amt"].replace([np.inf, -np.inf], np.nan)
        df["cust_avg_amt"] = df["cust_avg_amt"].fillna(df["transaction_amount"])
        df["cust_total_txn_log"] = np.log1p(df["cust_total_txn"])
        df["cust_total_amt_log"] = np.log1p(df["cust_total_amt"])
    else:
        for c in [
            "cust_total_txn",
            "cust_total_amt",
            "cust_avg_amt",
            "cust_total_txn_log",
            "cust_total_amt_log",
        ]:
            df[c] = 0.0

    if "ip_address" in df.columns:
        ip_cnt = df.groupby("ip_address")["transaction_amount"].transform("count")
        df["ip_txn_count"] = ip_cnt.astype(float)
        df["ip_txn_count_log"] = np.log1p(df["ip_txn_count"])
    else:
        df["ip_txn_count_log"] = 0.0

    if "device_used" in df.columns:
        dev_cnt = df.groupby("device_used")["transaction_amount"].transform("count")
        df["device_txn_count"] = dev_cnt.astype(float)
        df["device_txn_count_log"] = np.log1p(df["device_txn_count"])
    else:
        df["device_txn_count_log"] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df



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
        col_tr_te[hold_idx] = X_train.iloc[hold_idx][col].map(smooth).fillna(global_mean)

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



def run_high_perf_tree_ensemble(random_state=42):
    counter_stats = build_counterfeit_risk_features()
    df = load_and_feature_main(counter_stats)

    target_col = "is_fraudulent"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    y = df[target_col].astype(int)

    print_head("3) Prepare train/valid/test split")
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

    print_head("4) Encoding (Frequency + Target)")

    for c in cat_cols:
        freq = train[c].value_counts()
        train[c + "_freq"] = train[c].map(freq).astype(float)
        valid[c + "_freq"] = valid[c].map(freq).fillna(0).astype(float)
        test[c + "_freq"] = test[c].map(freq).fillna(0).astype(float)

    for c in cat_cols:
        train, valid, test, _ = add_target_encoding_kfold(
            train, y_train, valid, test, c, n_splits=5, smoothing=100
        )

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    print(f"[Features] numeric feature count: {len(numeric_cols)}")

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

    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[],
    )

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

    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    print_head("6) Predict & threshold tuning")

    valid_prob = 0.6 * lgbm.predict_proba(X_valid)[:, 1] + 0.4 * xgb.predict_proba(X_valid)[:, 1]
    test_prob = 0.6 * lgbm.predict_proba(X_test)[:, 1] + 0.4 * xgb.predict_proba(X_test)[:, 1]

    best_t, best_f1 = find_best_threshold(y_valid, valid_prob, start=0.1, end=0.9, step=0.01)
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

    return {
        "lgbm": lgbm,
        "xgb": xgb,
        "best_threshold": best_t,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "feature_names": numeric_cols,
    }

model_bundle = run_high_perf_tree_ensemble(random_state=42)
gc.collect();
