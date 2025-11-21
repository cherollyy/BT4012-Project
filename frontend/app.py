import streamlit as st
import numpy as np
import requests
import os
import re
import pandas as pd
import altair as alt
from pathlib import Path
# from geo import resolve_ip_batch, plot_country_choropleth
import seaborn as sns, matplotlib.pyplot as plt
import plotly.express as px
import time


API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="FraudGuard AI", layout="wide")
st.title("🛡️ FraudGuard AI Dashboard & Fraud Check")


# --- Model status helper -------------------------------------------------
def refresh_model_status():
    """Query backend /model_status and stash the result in session_state.

    Called on first load and when user clicks Retry.
    """
    try:
        r = requests.get(f"{API_URL}/model_status", timeout=4)
        if r.ok:
            data = r.json()
            st.session_state["model_loaded"] = bool(data.get("loaded", False))
            st.session_state["model_metadata"] = data.get("metadata")
            st.session_state["model_status_msg"] = data.get("message")
        else:
            st.session_state["model_loaded"] = False
            st.session_state["model_metadata"] = None
            st.session_state["model_status_msg"] = f"HTTP {r.status_code}"
    except Exception:
        st.session_state["model_loaded"] = False
        st.session_state["model_metadata"] = None
        st.session_state["model_status_msg"] = "Error contacting backend"


def try_rerun():
    """Safely attempt to rerun the Streamlit script.

    Some Streamlit builds do not expose `experimental_rerun`. Fall back to
    toggling a session_state flag and stopping the script which causes Streamlit
    to perform a rerun on the next user interaction.
    """
    try:
        # preferred in many Streamlit versions
        st.experimental_rerun()
    except Exception:
        # best-effort fallback
        st.session_state["_rerun_toggle"] = not st.session_state.get("_rerun_toggle", False)
        st.stop()


# ensure we check model status at least once per session
if "model_loaded" not in st.session_state:
    refresh_model_status()

# --- big button styles -------------------------------------------------
st.markdown(
    """
    <style>
    div.stButton > button {
        width: 100%;
        height: 70px;
        font-size: 22px;
        font-weight: 600;
    }
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    div.stButton {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# initialize page in session state if not present
if "page" not in st.session_state:
    st.session_state["page"] = "dashboard"

# render two large nav buttons side-by-side
col1, col2 = st.columns(2)
with col1:
    if st.button("📊 Dashboard", key="nav_dashboard", type="primary" if st.session_state["page"] == "dashboard" else "secondary", width="stretch"):
        st.session_state["page"] = "dashboard"
        st.rerun()
with col2:
    if st.button("🔍 Fraud Checker", key="nav_fraud", type="primary" if st.session_state["page"] == "fraud" else "secondary", width="stretch"):
        st.session_state["page"] = "fraud"
        st.rerun()

page = st.session_state["page"]


# -------------------------------
# IP → Country (lightweight version)
# -------------------------------
def ip_to_country(ip):
    """Simple placeholder converter."""
    if pd.isna(ip):
        return "Unknown"
    if isinstance(ip, str):
        # Private IP ranges → treat as internal
        if ip.startswith("192.") or ip.startswith("172.") or ip.startswith("10."):
            return "Local Network"
    return "Unknown"


# -------------------------------
# Load training / sample data
# -------------------------------
# def load_training_dataframe():
#     train_py = Path("backend/training/train_model.py")
#     df = None
#     discovered_path = None

#     if train_py.exists():
#         try:
#             txt = train_py.read_text()
#             m = re.search(r"RAW_DATA_PATH\s*=\s*[\"'](.+?)[\"']", txt)
#             if m:
#                 discovered_path = m.group(1)
#                 p = Path(discovered_path)
#                 if p.exists():
#                     df = pd.read_csv(p)
#                 else:
#                     # try repo-local data folder (several likely locations)
#                     alt_path = Path(discovered_path).name
#                     candidates = [
#                         Path("data") / alt_path,
#                         Path(__file__).resolve().parents[1] / "data" / alt_path,  # repo root /data
#                         Path("/app/data") / alt_path,  # container common mount
#                     ]
#                     for candidate in candidates:
#                         if candidate.exists():
#                             df = pd.read_csv(candidate)
#                             break
#         except Exception:
#             df = None

#     if df is None:
#         # try several fallback data directories for CSV matching the expected fraud dataset name
#         sample_candidates = [
#             Path("data"),
#             Path(__file__).resolve().parents[1] / "data",
#             Path("/app/data"),
#         ]
#         for base in sample_candidates:
#             if base.exists():
#                 for f in base.glob("Fraudulent_E-Commerce_Transaction_Data*.csv"):
#                     try:
#                         df = pd.read_csv(f)
#                         discovered_path = str(f)
#                         break
#                     except Exception:
#                         df = None
#                 if df is not None:
#                     break

#     return df, discovered_path





# ================================
# DASHBOARD PAGE
# ================================
if page == "dashboard":
    st.header("📊 Dashboard")
    st.markdown("Training data visualization and insights.")
    
    # Try multiple possible paths for the data file
    possible_paths = [
        Path("/app/data/Fraudulent_E-Commerce_Transaction_Data_2.csv"),  # Docker container mount
        Path(__file__).resolve().parent.parent / "data" / "Fraudulent_E-Commerce_Transaction_Data_2.csv",  # Local dev
    ]
    
    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            break
    
    if csv_path is None:
        st.error(f"❌ Data file not found. Tried paths:\n" + "\n".join(str(p) for p in possible_paths))
        st.stop()
    
    df = pd.read_csv(csv_path)
#--------------------
    # -------------------------------
    # Show Data + Charts (without preview/rows display)
    # -------------------------------
    if df is not None:
        # KPIs
        total_tx = len(df)
        total_fraud = int(df[df["Is Fraudulent"] == 1].shape[0])
        fraud_rate = total_fraud / max(1, total_tx)
        avg_amount = df["Transaction Amount"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total transactions", f"{total_tx:,}")
        col2.metric("Total frauds", f"{total_fraud:,}", delta=f"{fraud_rate*100:.2f}%")
        col3.metric("Avg amount", f"${avg_amount:,.2f}")


        #Overview of data -- Transaction Amount 
        st.subheader("Overview — Dataset Distribution")
        amount = "Transaction Amount"

        # ensure numeric
        df[amount] = pd.to_numeric(df[amount], errors="coerce")
        valid_count = int(df[amount].notna().sum())
        if valid_count == 0:
            st.warning(f"No numeric values found in '{amount}'.")
        else:
            st.write(f"Valid numeric rows for plotting: {valid_count:,}")

            # log toggle
            log_scale = st.checkbox("Log scale amount", value=False)

            # create sampled subset for performance (optional)
            sample_n = st.slider("Sample rows (0 = no sampling)", min_value=0, max_value=5000, value=2000, step=250)
            if sample_n and len(df) > sample_n:
                df_plot = df.dropna(subset=[amount]).sample(n=sample_n, random_state=42).reset_index(drop=True)
            else:
                df_plot = df.dropna(subset=[amount]).reset_index(drop=True)

            # produce transformed column for log plotting (use log1p to handle zeros)
            df_plot["log_amount"] = np.log1p(df_plot[amount].clip(lower=0))

            # choose x encoding depending on toggle
            if log_scale:
                x_enc = alt.X("log_amount:Q",
                            bin=alt.Bin(maxbins=60),
                            axis=alt.Axis(title="log(1 + Transaction Amount)"))
            else:
                x_enc = alt.X(f"{amount}:Q",
                            bin=alt.Bin(maxbins=60),
                            axis=alt.Axis(title="Transaction Amount"))

            chart = (
                alt.Chart(df_plot)
                .mark_bar()
                .encode(
                    x=x_enc,
                    y="count()",
                    tooltip=[alt.Tooltip(f"{amount}:Q"), "count()"]
                )
                .properties(height=350)
            )

            st.altair_chart(chart, use_container_width=True)


        # -----------------------------------------------
        # Identify fraud column
        # -----------------------------------------------
        # detect a fraud-like column (case-insensitive contains 'fraud')
        fraud_col = next((col for col in df.columns if 'fraud' in col.lower()), None)

        # -----------------------------------------------
        # Fraud vs Legit Transactions
        # -----------------------------------------------
        if fraud_col:
            st.subheader("Fraud vs Legit Transactions")
            fraud_counts = df[fraud_col].value_counts().reset_index()
            fraud_counts.columns = ["Class", "Count"]

            chart1 = (
                alt.Chart(fraud_counts)
                .mark_bar()
                .encode(
                    x=alt.X("Class:N", title="0 = Legit, 1 = Fraud"),
                    y="Count:Q",
                    color="Class:N",
                    tooltip=["Class", "Count"]
                )
                .properties(height=350)
            )
            st.altair_chart(chart1)

        # -----------------------------------------------
        # Transaction Amount Distribution
        # -----------------------------------------------
        amt_col = None
        for col in df.columns:
            if col.lower().startswith("transaction") and df[col].dtype != object:
                amt_col = col
                break

        if amt_col:
            st.subheader("Transaction Amount Distribution")
            chart2 = (
                alt.Chart(df)
                .mark_bar(opacity=0.7)
                .encode(
                    x=alt.X(f"{amt_col}:Q", bin=alt.Bin(maxbins=50)),
                    y="count()",
                    color=f"{fraud_col}:N" if fraud_col else alt.value("#4C78A8"),
                )
                .properties(height=350)
            )
            st.altair_chart(chart2)

        # -----------------------------------------------
        # Fraud Rate by Payment Method
        # -----------------------------------------------
        if "Payment Method" in df.columns and fraud_col:
            st.subheader("Fraud Rate by Payment Method")
            pm_df = (
                df.groupby("Payment Method")[fraud_col]
                .mean()
                .reset_index()
            )
            pm_df["FraudRate"] = pm_df[fraud_col] * 100

            chart3 = (
                alt.Chart(pm_df)
                .mark_bar()
                .encode(
                    x="Payment Method:N",
                    y=alt.Y("FraudRate:Q", title="Fraud Rate (%)"),
                    color="Payment Method:N",
                    tooltip=["Payment Method", "FraudRate"]
                )
                .properties(height=350)
            )
            st.altair_chart(chart3)


        # -----------------------------------------------
        # Time series: daily transactions & frauds with rolling average
        # -----------------------------------------------
        st.subheader("Daily Transactions & Frauds with Rolling Average")
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
        daily = df.set_index("Transaction Date").resample("D").agg({"Transaction ID":"count", "Is Fraudulent":"sum"})
        daily.columns = ["transactions","frauds"]
        daily["fraud_rate"] = 100 * daily["frauds"] / daily["transactions"].replace(0,1)
        daily["transactions_ma7"] = daily["transactions"].rolling(7).mean()

        import plotly.express as px
        fig = px.line(daily, y=["transactions","transactions_ma7"], labels={"value":"Count","index":"Date"})
        fig.update_layout(title="Daily transactions (and 7-day MA)")
        st.plotly_chart(fig)


        # -----------------------------------------------
        # Hour-of-day vs Day-of-week heatmap (when frauds spike)
        # -----------------------------------------------
        st.subheader("Fraud Heatmap by Hour of Day and Day of Week (dow)")
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
        df["hour"] = df["Transaction Date"].dt.hour
        df["dow"] = df["Transaction Date"].dt.day_name()

        heat = (
            df[df["Is Fraudulent"] == 1]
            .groupby(["dow","hour"])
            .size()
            .reset_index(name="fraud_count")
        )

        # pivot so rows=dow in order Mon..Sun
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heat_piv = heat.pivot(index="dow", columns="hour", values="fraud_count").reindex(order).fillna(0)

        fig, ax = plt.subplots(figsize=(14,3))
        sns.heatmap(heat_piv, cmap="Reds", ax=ax)
        ax.set_xlabel("Hour of day")
        st.pyplot(fig)


        # -----------------------------------------------
        # Fraud Rate by Device Type
        # -----------------------------------------------
        if "Device Used" in df.columns and fraud_col:
            st.subheader("Fraud Rate by Device Type")
            ddf = df.groupby("Device Used")[fraud_col].mean().reset_index()
            ddf["FraudRate"] = ddf[fraud_col] * 100

            chart4 = (
                alt.Chart(ddf)
                .mark_bar()
                .encode(
                    x="Device Used:N",
                    y="FraudRate:Q",
                    color="Device Used:N",
                    tooltip=["Device Used", "FraudRate"]
                )
                .properties(height=350)
            )
            st.altair_chart(chart4)

        # -----------------------------------------------
        # Product Categories Distribution Among Fraud Cases
        # -----------------------------------------------
        st.subheader("Product Categories in Fraud Cases")
        cat = (df[df["Is Fraudulent"] == 1]
        .groupby("Product Category")
        .size()
        .reset_index(name="fraud_count")
        .sort_values("fraud_count", ascending=False).head(20))
        chart = alt.Chart(cat).mark_bar().encode(
            x="fraud_count:Q",
            y=alt.Y("Product Category:N", sort='-x'),
            tooltip=["Product Category","fraud_count"]
        ).properties(height=600)
        st.altair_chart(chart)


        # -----------------------------------------------
        # Customer Age Distribution Among Fraud Cases
        # -----------------------------------------------
        if "Customer Age" in df.columns and fraud_col:
            st.subheader("Customer Age in Fraud Cases")
            fraud_only = df[df[fraud_col] == 1]

            chart5 = (
                alt.Chart(fraud_only)
                .mark_bar()
                .encode(
                    x="Customer Age:N",
                    y="count()",
                    color=alt.value("#4C78A8"),
                )
                .properties(height=350)
            )
            st.altair_chart(chart5)    

        # -----------------------------------------------
        # Outlier scatterplot 
        # -----------------------------------------------
        st.subheader("Amount vs Account Age (sample)")
        # sample safely, drop rows with missing numeric values, make sure column names match exactly
        sample_n = min(len(df), 2000)
        df_sample = (
            df
            .dropna(subset=["Account Age Days", "Transaction Amount"])      # remove rows missing the axes
            .sample(n=sample_n, random_state=42)                           # sample reproducibly
            .reset_index(drop=True)
        )
        # ensure fraud column is string/categorical so colors render properly
        df_sample["Is Fraudulent"] = df_sample["Is Fraudulent"].astype(str)
        # now pass the sampled DF and the column name (not a full-length series)
        import plotly.express as px
        fig = px.scatter(
            df_sample,
            x="Account Age Days",
            y="Transaction Amount",
            color="Is Fraudulent",               
            hover_data=["Customer ID", "Transaction ID"],
            title="Amount vs Account Age (sample)"
        )
        st.plotly_chart(fig)


# ================================
# FRAUD CHECKER PAGE
# ================================
elif page == "fraud":
    st.header("🔍 Fraud Checker")
    st.info("Single-check predictions are approximate — if you can provide historical/proxy values (customer totals, ip counts, amount per unit) add them in Advanced (optional) to improve accuracy.")
    # show model readiness status
    if st.session_state.get("model_loaded"):
        st.success("Model loaded and ready to predict")
    else:
        # show progress bar while polling model status; this is a friendly UX to indicate
        # background training / model loading. We poll /model_status repeatedly for up
        # to MODEL_WAIT_TIMEOUT seconds (default 120s) and update a progress bar.
        st.info("Model not loaded — attempting to detect training progress. Predictions are disabled until the model loads.")

        # place holders for progress UI
        progress_placeholder = st.empty()
        prog = progress_placeholder.progress(0)

        start = time.time()
        max_wait = int(os.getenv("MODEL_WAIT_TIMEOUT", "120"))
        pct = 0

        # Active polling loop: query /train_status and update the progress bar
        while time.time() - start < max_wait:
            try:
                r = requests.get(f"{API_URL}/train_status", timeout=4)
                if r.ok:
                    ts = r.json()
                    in_prog = bool(ts.get("in_progress"))
                    pct_remote = int(ts.get("percent") or 0)
                    msg = ts.get("message")

                    if in_prog:
                        # show remote percent (cap to 95 while waiting for final save)
                        display_pct = max(0, min(95, pct_remote))
                        prog.progress(display_pct)
                        progress_placeholder.info(msg or f"Training in progress ({display_pct}%)")
                    else:
                        # if remote says not in progress, check model_status to see if it's loaded
                        r2 = requests.get(f"{API_URL}/model_status", timeout=4)
                        if r2.ok:
                            ms = r2.json()
                            if ms.get("loaded"):
                                prog.progress(100)
                                st.session_state["model_loaded"] = True
                                st.session_state["model_metadata"] = ms.get("metadata")
                                progress_placeholder.success("Model loaded and ready to predict")
                                break
                            else:
                                progress_placeholder.error("Model not loaded yet. Start training in the backend or try again.")
                                break
                else:
                    # fallback: check model_status directly
                    r2 = requests.get(f"{API_URL}/model_status", timeout=4)
                    if r2.ok and r2.json().get("loaded"):
                        prog.progress(100)
                        st.session_state["model_loaded"] = True
                        st.session_state["model_metadata"] = r2.json().get("metadata")
                        progress_placeholder.success("Model loaded and ready to predict")
                        break
            except Exception:
                # ignore transient errors
                pass

            # advance a bit to show activity while waiting
            pct = min(95, pct + 5)
            prog.progress(pct)
            time.sleep(2)

        if not st.session_state.get("model_loaded"):
            progress_placeholder.error("Model still not loaded after waiting. Click Retry or ask an admin to train the model.")

        # retry button (keeps same behaviour as before)
        colr, colc = st.columns([3,1])
        with colr:
            if st.button("Retry model status", key="retry_header"):
                refresh_model_status()
                try_rerun()
        with colc:
            st.write(" ")

    user_tab, tx_tab = st.tabs(["User Fraud Check", "Transaction Fraud Check"])

    def build_features_from_inputs(kind: str, feature_names: list):
        """Best-effort mapping from UI inputs to model feature names.

        kind: 'user' or 'tx'
        feature_names: list of strings from metadata (may contain spaces/caps)
        """
        features = {}
        # helper to find a value from session state with fallback
        def safe(key, default=0):
            return st.session_state.get(key, default)

        for fname in (feature_names or []):
            low = fname.lower()
            # transaction amount
            if "amount" in low:
                if kind == "tx":
                    features[fname] = float(safe("amount_input", 0.0) or 0.0)
                else:
                    features[fname] = float(safe("avg_order_input", 0.0) or 0.0)
            # quantity / units
            elif "quantity" in low or "qty" in low:
                if kind == "tx":
                    features[fname] = int(safe("cust_total_txn_tx", 1) or 1)
                else:
                    features[fname] = int(safe("total_tx_input", 1) or 1)
            # customer / customer age
            elif "customer" in low and "age" in low:
                if kind == "tx":
                    features[fname] = int(safe("customer_age_input", 0) or 0)
                else:
                    features[fname] = int(safe("age_input", 0) or 0)
            # account age
            elif "account" in low and "age" in low:
                features[fname] = int(safe("account_age_input", 0) or 0)
            # transaction hour
            elif "hour" in low:
                features[fname] = int(safe("transaction_hour_input", 12) or 12)
            else:
                # fallback: try some commonly available inputs
                if kind == "tx":
                    features[fname] = float(safe("amount_input", 0.0) or 0.0)
                else:
                    features[fname] = float(safe("avg_order_input", 0.0) or 0.0)

        return features

    with user_tab:
        st.subheader("👤 User Information Check")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age *", 18, 100, 30, key="age_input")
            past_fraud = st.number_input("Past Fraud Cases", 0, 50, 0, key="past_fraud_input")
            avg_order = st.number_input("Average Order Value ($)", 0.0, 50000.0, 150.0, key="avg_order_input")
        with col2:
            account_age = st.number_input("Account Age (days)", 0, 5000, 365, help="Number of days since account creation", key="account_age_input")
            total_tx = st.number_input("Total Transactions", 0, 5000, 15, help="Total number of transactions by this user (optional)", key="total_tx_input")

        # Advanced optional inputs for better single-row inference
        with st.expander("Advanced (optional): provide proxies for historical aggregates"):
            cust_total_txn = st.number_input("Customer total transactions (proxy)", 0, 100000, 0, help="Approximate total transactions for this customer; helps model estimate customer history", key="cust_total_txn_user")
            cust_avg_amt = st.number_input("Customer average order value ($)", 0.0, 1e7, 0.0, help="Approx average order value for this customer", key="cust_avg_amt_user")
            ip_txn_count = st.number_input("IP transaction count (proxy)", 0, 100000, 0, help="Approx number of transactions from this IP address", key="ip_txn_count_user")
            
        if not st.session_state.get("model_loaded", False):
            st.warning("Model not loaded — predictions disabled. Retry or ask an admin to train the model.")
            if st.button("Retry model status", key="retry_user"):
                refresh_model_status()
                try_rerun()
        else:
            if st.button("🔍 Analyze User Fraud Risk", width='stretch',key="analyze_user"):
                # prefer to build a 'features' dict matching model metadata when available
                meta = st.session_state.get("model_metadata") or {}
                fns = meta.get("feature_names") if isinstance(meta, dict) else None
                features = build_features_from_inputs("user", fns)

                # include optional advanced inputs only when provided (non-zero)
                if cust_total_txn and cust_total_txn > 0:
                    features.setdefault("cust_total_txn", int(cust_total_txn))
                if cust_avg_amt and cust_avg_amt > 0:
                    features.setdefault("cust_avg_amt", float(cust_avg_amt))
                if ip_txn_count and ip_txn_count > 0:
                    features.setdefault("ip_txn_count", int(ip_txn_count))

                payload = {"features": features}
                try:
                    r = requests.post(f"{API_URL}/predict_fraud", json=payload, timeout=15)
                    if r.status_code == 200:
                        j = r.json()
                        st.success(f"Fraud Probability: {j.get('fraud_probability')}%")
                    else:
                        st.error(f"API error (HTTP {r.status_code}): {r.text}")
                except Exception as e:
                    st.error(f"API error: {e}")

    with tx_tab:
        st.subheader("💳 Transaction Details Check")
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount ($)", 0.0, 100000.0, 299.99, key="amount_input")
            payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"])
            device = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet"])
        with col2:
            ip = st.text_input("IP Address", help="Optional - can help detect suspicious IPs")
            browser = st.selectbox("Browser Type", ["Chrome", "Firefox", "Safari", "Edge"], help="Browser used for transaction")
            shipping = st.text_input("Shipping Address")
            billing = st.text_input("Billing Address")

        # Advanced optional inputs for transaction
        with st.expander("Advanced (optional): provide proxies for aggregates"):
            amount_per_unit = st.number_input("Amount per unit ($)", 0.0, 1e7, 0.0, help="Transaction amount divided by quantity (if applicable)", key="amount_per_unit_tx")
            cust_total_txn_tx = st.number_input("Customer total transactions (proxy)", 0, 100000, 0, help="Approx total transactions for this customer", key="cust_total_txn_tx")
            ip_txn_count_tx = st.number_input("IP transaction count (proxy)", 0, 100000, 0, help="Approx number of transactions from this IP address", key="ip_txn_count_tx")

        transaction_hour = st.number_input("Transaction Hour (0-23)", 0, 23, 12, key="transaction_hour_input")
        customer_age = st.number_input("Customer Age", 0, 120, 30, key="customer_age_input")

        if not st.session_state.get("model_loaded", False):
            st.warning("Model not loaded — predictions disabled. Retry or ask an admin to train the model.")
            if st.button("Retry model status", key="retry_tx"):
                refresh_model_status()
                try_rerun()
        else:
            if st.button("🔍 Analyze Transaction Fraud Risk", width='stretch', key="analyze_tx"):
                # send a transaction-style payload (user-friendly fields) so backend can map them
                payload = {
                    "transaction": {
                        "amount": float(amount),
                        "payment_method": payment_method,
                        "device": device,
                        "ip": ip,
                        "browser": browser,
                        "shipping": shipping,
                        "billing": billing,
                        "transaction_hour": int(transaction_hour),
                        "age": int(customer_age),
                    }
                }

                # include optional advanced inputs only when provided
                if amount_per_unit and amount_per_unit > 0:
                    payload["transaction"]["amount_per_unit"] = float(amount_per_unit)
                if cust_total_txn_tx and cust_total_txn_tx > 0:
                    payload["transaction"]["cust_total_txn"] = int(cust_total_txn_tx)
                if ip_txn_count_tx and ip_txn_count_tx > 0:
                    payload["transaction"]["ip_txn_count"] = int(ip_txn_count_tx)

                try:
                    r = requests.post(f"{API_URL}/predict_fraud", json=payload, timeout=15)
                    if r.status_code == 200:
                        j = r.json()
                        st.success(f"Fraud Probability: {j.get('fraud_probability')}%")
                    else:
                        st.error(f"API error (HTTP {r.status_code}): {r.text}")
                except Exception as e:
                    st.error(f"API error: {e}")

else:
    st.session_state["page"] = "dashboard"
    st.rerun()