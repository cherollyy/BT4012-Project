import os, gc
import numpy as np
import pandas as pd
from pathlib import Path

current_file_path = Path(__file__).resolve()

ROOT = current_file_path.parent.parent.parent

def safe_lower_strip(series):
    return series.astype(str).str.lower().str.strip()

def safe_int_from_codes(series, fill_value=0):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.fillna(fill_value)
    return s.astype(int)

def build_counterfeit_risk_features():
    print("Building counterfeit stats...")
    df_txn = pd.read_csv(Path(ROOT, "data/_counterfeit_transactions.csv"))
    df_prod = pd.read_csv(Path(ROOT, "data/counterfeit_products.csv"))

    df_txn["payment_method_norm"] = safe_lower_strip(df_txn["payment_method"])
    if "customer_location" in df_txn.columns:
        df_txn["customer_location_norm"] = safe_lower_strip(df_txn["customer_location"])

    stats_txn_pm = (
        df_txn.groupby("payment_method_norm")["involves_counterfeit"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "cnt_txn_counterfeit_rate_pm", "count": "cnt_txn_count_pm"})
    )

    if "customer_location_norm" in df_txn.columns:
        stats_txn_loc_pm = (
            df_txn.groupby(["customer_location_norm", "payment_method_norm"])["involves_counterfeit"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={
                "mean": "cnt_txn_counterfeit_rate_loc_pm",
                "count": "cnt_txn_count_loc_pm",
            })
        )
    else:
        stats_txn_loc_pm = None

    df_prod["category_norm"] = safe_lower_strip(df_prod["category"])
    stats_prod_cat = (
        df_prod.groupby("category_norm")["is_counterfeit"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={
            "mean": "cnt_prod_counterfeit_rate_cat",
            "count": "cnt_prod_count_cat",
        })
    )

    return {
        "stats_txn_pm": stats_txn_pm,
        "stats_txn_loc_pm": stats_txn_loc_pm,
        "stats_prod_cat": stats_prod_cat,
    }


def preprocessing_only(counter_stats):
    print("Loading main E-commerce dataset...")
    df = pd.read_csv(os.path.join(ROOT, "data/Fraudulent_E-Commerce_Transaction_Data_FULL.csv"))

    df = df.rename(columns=lambda c: c.strip().replace(" ", "_").replace("-", "_"))
    rename_map = {
        "Transaction_Amount": "transaction_amount",
        "Transaction_Date": "transaction_date",
        "Customer_ID": "customer_id",
        "Customer_Age": "customer_age",
        "Customer_Location": "customer_location",
        "Payment_Method": "payment_method",
        "Product_Category": "product_category",
        "Device_Used": "device_used",
        "IP_Address": "ip_address",
        "Shipping_Address": "shipping_address",
        "Billing_Address": "billing_address",
        "Account_Age_Days": "account_age_days",
        "Transaction_Hour": "transaction_hour",
        "Is_Fraudulent": "is_fraudulent",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["payment_method", "product_category", "customer_location"]:
        if col in df.columns:
            df[col + "_norm"] = safe_lower_strip(df[col])
    s = counter_stats

    if "payment_method_norm" in df.columns:
        df = df.merge(s["stats_txn_pm"], how="left", on="payment_method_norm")

    if s["stats_txn_loc_pm"] is not None:
        if "customer_location_norm" in df.columns:
            df = df.merge(
                s["stats_txn_loc_pm"],
                how="left",
                on=["customer_location_norm", "payment_method_norm"],
            )

    if "product_category_norm" not in df.columns:
        df["product_category_norm"] = safe_lower_strip(df.get("product_category", ""))

    df = df.merge(
        s["stats_prod_cat"],
        how="left",
        left_on="product_category_norm",
        right_on="category_norm",
    )

    df = df.fillna(0)

    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["tx_dayofweek"] = df["transaction_date"].dt.weekday
        df["tx_day"] = df["transaction_date"].dt.day
        df["tx_month"] = df["transaction_date"].dt.month
    else:
        df["tx_dayofweek"] = -1
        df["tx_day"] = -1
        df["tx_month"] = -1

    # Hour
    if "transaction_hour" in df.columns:
        hour = pd.to_numeric(df["transaction_hour"], errors="coerce").fillna(-1).astype(int)
        hour_clip = hour.clip(0, 23)
        df["hour_sin"] = np.sin(2 * np.pi * hour_clip / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour_clip / 24)
        df["is_night"] = ((hour_clip <= 6) | (hour_clip >= 22)).astype(int)
    else:
        df["hour_sin"] = 0
        df["hour_cos"] = 0
        df["is_night"] = 0

    df["is_weekend"] = df["tx_dayofweek"].isin([5, 6]).astype(int)

    if "transaction_amount" in df.columns:
        df["transaction_amount"] = df["transaction_amount"].fillna(0)
        df["log_amount"] = np.log1p(df["transaction_amount"])
        high_th = df["transaction_amount"].quantile(0.95)
        df["is_high_amount"] = (df["transaction_amount"] >= high_th).astype(int)
    else:
        df["log_amount"] = 0
        df["is_high_amount"] = 0

    if "customer_id" in df.columns:
        df["cust_txn_count"] = df.groupby("customer_id")["transaction_amount"].transform("count")
        df["cust_txn_sum"] = df.groupby("customer_id")["transaction_amount"].transform("sum")
        df["cust_txn_mean"] = df["cust_txn_sum"] / df["cust_txn_count"].replace(0, np.nan)
        df["cust_txn_mean"] = df["cust_txn_mean"].fillna(0)
        df["cust_txn_count_log"] = np.log1p(df["cust_txn_count"])
    else:
        df["cust_txn_mean"] = 0

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

stats = build_counterfeit_risk_features()
df_processed = preprocessing_only(stats)

SAVE_PATH = os.path.join(ROOT, "ECommerce_preprocessed_cleaned.csv")
df_processed.to_csv(SAVE_PATH, index=False)

print("Saved processed file to:", SAVE_PATH)
