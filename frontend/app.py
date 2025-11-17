import streamlit as st
import requests
import os
import re
import pandas as pd
import altair as alt
from pathlib import Path

API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="FraudGuard AI", layout="wide")
st.title("🛡️ FraudGuard AI Dashboard & Fraud Check")

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
    if st.button("📊 Dashboard", key="nav_dashboard", type="primary" if st.session_state["page"] == "dashboard" else "secondary", use_container_width=True):
        st.session_state["page"] = "dashboard"
        st.rerun()
with col2:
    if st.button("🔍 Fraud Checker", key="nav_fraud", type="primary" if st.session_state["page"] == "fraud" else "secondary", use_container_width=True):
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
def load_training_dataframe():
    train_py = Path("backend/training/train_model.py")
    df = None
    discovered_path = None

    if train_py.exists():
        try:
            txt = train_py.read_text()
            m = re.search(r"RAW_DATA_PATH\s*=\s*[\"'](.+?)[\"']", txt)
            if m:
                discovered_path = m.group(1)
                p = Path(discovered_path)
                if p.exists():
                    df = pd.read_csv(p)
                else:
                    # try repo-local data folder (several likely locations)
                    alt_path = Path(discovered_path).name
                    candidates = [
                        Path("data") / alt_path,
                        Path(__file__).resolve().parents[1] / "data" / alt_path,  # repo root /data
                        Path("/app/data") / alt_path,  # container common mount
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            df = pd.read_csv(candidate)
                            break
        except Exception:
            df = None

    if df is None:
        # try several fallback data directories for CSV matching the expected fraud dataset name
        sample_candidates = [
            Path("data"),
            Path(__file__).resolve().parents[1] / "data",
            Path("/app/data"),
        ]
        for base in sample_candidates:
            if base.exists():
                for f in base.glob("Fraudulent_E-Commerce_Transaction_Data*.csv"):
                    try:
                        df = pd.read_csv(f)
                        discovered_path = str(f)
                        break
                    except Exception:
                        df = None
                if df is not None:
                    break

    return df, discovered_path


# ================================
# DASHBOARD PAGE
# ================================
if page == "dashboard":
    st.header("📊 Dashboard")
    st.markdown("Training data visualization and insights.")

    df, discovered_path = load_training_dataframe()

    # -------------------------------
    # Show Data + Charts (without preview/rows display)
    # -------------------------------
    if df is not None:
        # -----------------------------------------------
        # Overview Distribution Chart
        # -----------------------------------------------
        st.subheader("Overview — Dataset Distribution")
        
        # Select a representative numeric column for the overview
        overview_col = None
        if "fraud_probability" in df.columns:
            overview_col = "fraud_probability"
        else:
            # try common amount-like names
            for candidate in ["amount", "transaction_amount", "TransactionAmt", "TransactionAmount"]:
                if candidate in df.columns:
                    overview_col = candidate
                    break
        
        if overview_col is None:
            # fallback to first numeric column
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if num_cols:
                overview_col = num_cols[0]

        if overview_col is not None:
            hist = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(f"{overview_col}:Q", bin=alt.Bin(maxbins=40)),
                    y="count()",
                    tooltip=[f"{overview_col}:Q", "count()"]
                )
                .properties(height=400)
            )
            st.altair_chart(hist, use_container_width=True)

        # -----------------------------------------------
        # Identify fraud column
        # -----------------------------------------------
        fraud_col = None
        for col in ["IsFraud", "fraud", "Fraud", "is_fraud"]:
            if col in df.columns:
                fraud_col = col
                break

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
            st.altair_chart(chart1, use_container_width=True)

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
            st.altair_chart(chart2, use_container_width=True)

        # -----------------------------------------------
        # Fraud Rate by Payment Method
        # -----------------------------------------------
        if "PaymentMethod" in df.columns and fraud_col:
            st.subheader("Fraud Rate by Payment Method")
            pm_df = (
                df.groupby("PaymentMethod")[fraud_col]
                .mean()
                .reset_index()
            )
            pm_df["FraudRate"] = pm_df[fraud_col] * 100

            chart3 = (
                alt.Chart(pm_df)
                .mark_bar()
                .encode(
                    x="PaymentMethod:N",
                    y=alt.Y("FraudRate:Q", title="Fraud Rate (%)"),
                    color="PaymentMethod:N",
                    tooltip=["PaymentMethod", "FraudRate"]
                )
                .properties(height=350)
            )
            st.altair_chart(chart3, use_container_width=True)

        # -----------------------------------------------
        # Fraud Rate by Device Type
        # -----------------------------------------------
        if "Device" in df.columns and fraud_col:
            st.subheader("Fraud Rate by Device Type")
            ddf = df.groupby("Device")[fraud_col].mean().reset_index()
            ddf["FraudRate"] = ddf[fraud_col] * 100

            chart4 = (
                alt.Chart(ddf)
                .mark_bar()
                .encode(
                    x="Device:N",
                    y="FraudRate:Q",
                    color="Device:N",
                    tooltip=["Device", "FraudRate"]
                )
                .properties(height=350)
            )
            st.altair_chart(chart4, use_container_width=True)

        # -----------------------------------------------
        # Browser Distribution Among Fraud Cases
        # -----------------------------------------------
        if "Browser" in df.columns and fraud_col:
            st.subheader("Browser Types in Fraud Cases")
            fraud_only = df[df[fraud_col] == 1]

            chart5 = (
                alt.Chart(fraud_only)
                .mark_bar()
                .encode(
                    x="Browser:N",
                    y="count()",
                    color="Browser:N",
                )
                .properties(height=350)
            )
            st.altair_chart(chart5, use_container_width=True)

        # -----------------------------------------------
        # Singapore CPI Trend (Placeholder)
        # -----------------------------------------------
        st.subheader("📈 Singapore CPI Trend (Placeholder)")
        cpi_placeholder = pd.DataFrame({
            "Month": pd.date_range("2023-01-01", periods=6, freq="M"),
            "CPI": [101, 102, 102.5, 103, 103.7, 104]
        }).set_index("Month")
        st.line_chart(cpi_placeholder)

        # -----------------------------------------------
        # Fraud Heatmap by Country (if country data available)
        # -----------------------------------------------
        # Try to find country or location-like columns
        country_col = None
        for c in ["Country", "country", "COUNTRY", "country_name", "Customer Location", "customer_location", "Location", "location"]:
            if c in df.columns:
                country_col = c
                break
        
        # Also try case-insensitive search
        if country_col is None:
            for col in df.columns:
                if col.lower() in ["country", "location", "customer location"]:
                    country_col = col
                    break

        ip_col = None
        for c in ["IP Address", "IP", "ip", "Ip", "client_ip", "ip_address"]:
            if c in df.columns:
                ip_col = c
                break

        if country_col and fraud_col:
            st.subheader("🌍 Fraud Heatmap by Country/Location")
            country_fraud_df = (
                df[df[fraud_col] == 1]
                .groupby(country_col)
                .size()
                .reset_index(name="FraudCount")
            )
            
            if len(country_fraud_df) > 0:
                st.write(f"Fraud cases by {country_col}:")
                st.bar_chart(country_fraud_df.set_index(country_col)["FraudCount"])
        elif ip_col and fraud_col:
            st.subheader("🌍 Fraud Heatmap by Country (IP Geolocation)")
            st.info("This feature resolves IP addresses to countries and displays a geographic heatmap. Click the button below to start.")
            
            if st.button("Resolve IPs to Countries (external API)", use_container_width=True):
                # Resolve unique IPs (limited batch to avoid rate limits)
                unique_ips = df[ip_col].dropna().astype(str).unique().tolist()
                # Filter out invalid IPs
                unique_ips = [ip for ip in unique_ips if ip and ip.lower() != 'nan']
                unique_ips = unique_ips[:200]  # limit to 200
                
                if len(unique_ips) == 0:
                    st.warning("No valid IP addresses found in the dataset.")
                else:
                    ip_to_country_map = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, ip in enumerate(unique_ips):
                        status_text.text(f"Resolving IP {idx+1}/{len(unique_ips)}: {ip}")
                        try:
                            r = requests.get(f"http://ip-api.com/json/{ip}?fields=country", timeout=2).json()
                            country = r.get("country")
                            ip_to_country_map[ip] = country if country else "Unknown"
                        except Exception:
                            ip_to_country_map[ip] = "Unknown"
                        
                        progress_bar.progress((idx + 1) / len(unique_ips))
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Map resolved countries back to dataframe
                    df["_resolved_country"] = df[ip_col].astype(str).map(ip_to_country_map).fillna("Unknown")
                    
                    country_fraud_df = (
                        df[df[fraud_col] == 1]
                        .groupby("_resolved_country")
                        .size()
                        .reset_index(name="FraudCount")
                    )
                    
                    if len(country_fraud_df) > 0:
                        st.write("Fraud cases by country (from IP geolocation):")
                        st.bar_chart(country_fraud_df.set_index("_resolved_country")["FraudCount"])

    else:
        st.warning("⚠️ No training data found. Please ensure the dataset is available in `data/` or mounted in the container at `/app/data/`.")


# ================================
# FRAUD CHECKER PAGE
# (UNCHANGED)
# ================================
elif page == "fraud":
    st.header("🔍 Fraud Checker")
    user_tab, tx_tab = st.tabs(["User Fraud Check", "Transaction Fraud Check"])

    with user_tab:
        st.subheader("👤 User Information Check")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age *", 18, 100, 30)
            past_fraud = st.number_input("Past Fraud Cases", 0, 50, 0)
        with col2:
            account_age = st.number_input("Account Age (days) *", 0, 5000, 365)
            total_tx = st.number_input("Total Transactions *", 0, 5000, 15)
            avg_order = st.number_input("Average Order Value ($)", 0.0, 50000.0, 150.0)

        if st.button("🔍 Analyze User Fraud Risk", use_container_width=True):
            data = {
                "age": age,
                "account_age": account_age,
                "total_transactions": total_tx,
                "past_fraud": past_fraud,
                "avg_order_value": avg_order,
            }
            try:
                r = requests.post(f"{API_URL}/predict_user", json=data, timeout=5).json()
                st.success(f"Fraud Probability: {r['fraud_probability']}%")
                st.write(f"Risk Level: **{r['risk_level']}**")
            except Exception as e:
                st.error(f"API error: {e}")

    with tx_tab:
        st.subheader("💳 Transaction Details Check")
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount ($) *", 0.0, 100000.0, 299.99)
            payment_method = st.selectbox("Payment Method *", ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"])
            device = st.selectbox("Device Type *", ["Desktop", "Mobile", "Tablet"])
        with col2:
            ip = st.text_input("IP Address")
            browser = st.selectbox("Browser Type", ["Chrome", "Firefox", "Safari", "Edge"])
            shipping = st.text_input("Shipping Address *")
            billing = st.text_input("Billing Address")

        if st.button("🔍 Analyze Transaction Fraud Risk", use_container_width=True):
            data = {
                "amount": amount,
                "payment_method": payment_method,
                "device": device,
                "ip": ip,
                "browser": browser,
                "shipping": shipping,
                "billing": billing,
            }
            try:
                r = requests.post(f"{API_URL}/predict_transaction", json=data, timeout=5).json()
                st.success(f"Fraud Probability: {r['fraud_probability']}%")
                st.write(f"Risk Level: **{r['risk_level']}**")
            except Exception as e:
                st.error(f"API error: {e}")

else:
    st.session_state["page"] = "dashboard"
    st.rerun()




