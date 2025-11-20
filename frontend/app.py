import streamlit as st
import requests
import os
import re
import pandas as pd
import altair as alt
from pathlib import Path
from geo import resolve_ip_batch, plot_country_choropleth
import seaborn as sns, matplotlib.pyplot as plt
import plotly.express as px


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
                for f in base.glob("Fraudulent_E-Commerce_Transaction_Data_2.csv"):
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

    # <-- CALL the loader here -->
    df, discovered_path = load_training_dataframe()
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
        log_scale = st.checkbox("Log scale amount", value=False)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{amount}:Q", bin=alt.Bin(maxbins=60), scale=alt.Scale(type='log') if log_scale else alt.Scale()),
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
            st.altair_chart(chart3, use_container_width=True)


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
        st.plotly_chart(fig, use_container_width=True)


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
            st.altair_chart(chart4, use_container_width=True)

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
        st.altair_chart(chart, use_container_width=True)


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
            st.altair_chart(chart5, use_container_width=True)    

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
        st.plotly_chart(fig, use_container_width=True)


        # -----------------------------------------------
        # Singapore CPI Trend (Placeholder)
        # -----------------------------------------------
        '''
        st.subheader("📈 Singapore CPI Trend (Placeholder)")
        cpi_placeholder = pd.DataFrame({
            "Month": pd.date_range("2023-01-01", periods=6, freq="M"),
            "CPI": [101, 102, 102.5, 103, 103.7, 104]
        }).set_index("Month")
        st.line_chart(cpi_placeholder)'''

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
            st.info("This feature resolves IP addresses to countries and displays a geographic heatmap. Use the buttons below to resolve top fraud IPs or all unique IPs (careful with API limits).")

            left_col, right_col = st.columns([2,1])

            with left_col:
                if st.button("Resolve top fraud IPs (recommended)", use_container_width=True):
                    # Resolve only IPs from fraud cases, starting with most frequent
                    fraud_ips_series = df.loc[df[fraud_col] == 1, ip_col].dropna().astype(str)
                    freq = fraud_ips_series.value_counts()
                    top_n = 500  # configurable: smaller = safer for free API
                    top_ips = freq.head(top_n).index.tolist()

                    if len(top_ips) == 0:
                        st.warning("No IPs found among fraud cases.")
                    else:
                        with st.spinner(f"Resolving up to {len(top_ips)} IPs..."):
                            ip_to_country = resolve_ip_batch(top_ips, max_per_run=top_n)
                        df["_resolved_country"] = df[ip_col].astype(str).map(lambda ip: ip_to_country.get(ip))
                        df["_resolved_country"] = df["_resolved_country"].fillna("Unknown")

                        country_fraud_df = (
                            df[df[fraud_col] == 1]
                            .groupby("_resolved_country")
                            .size()
                            .reset_index(name="FraudCount")
                        )

                        st.dataframe(country_fraud_df.sort_values("FraudCount", ascending=False).head(200))
                        plot_country_choropleth(country_fraud_df, country_col="_resolved_country", count_col="FraudCount")

            with right_col:
                if st.button("Resolve ALL unique IPs (dangerous)", use_container_width=True):
                    unique_ips = df[ip_col].dropna().astype(str).unique().tolist()
                    max_allowed = 2000
                    if len(unique_ips) > max_allowed:
                        st.warning(f"Too many unique IPs ({len(unique_ips)}). Use the 'top' option or increase max_per_run with caution.")
                    else:
                        with st.spinner(f"Resolving {len(unique_ips)} IPs..."):
                            ip_to_country = resolve_ip_batch(unique_ips, max_per_run=len(unique_ips))
                        df["_resolved_country"] = df[ip_col].astype(str).map(lambda ip: ip_to_country.get(ip))
                        df["_resolved_country"] = df["_resolved_country"].fillna("Unknown")

                        country_fraud_df = (
                            df[df[fraud_col] == 1]
                            .groupby("_resolved_country")
                            .size()
                            .reset_index(name="FraudCount")
                        )

                        st.dataframe(country_fraud_df.sort_values("FraudCount", ascending=False).head(200))
                        plot_country_choropleth(country_fraud_df, country_col="_resolved_country", count_col="FraudCount")

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