import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config("Medical Claims AI Dashboard", layout="wide")

# ======================================================
# DEFAULT CPI TABLE (Fallback)
# ======================================================
DEFAULT_CPI = {
    2018: 120.0,
    2019: 121.8,
    2020: 123.1,
    2021: 125.4,
    2022: 128.6,
    2023: 131.2,
    2024: 134.0,
    2025: 136.5
}

# ======================================================
# CPI LOADERS
# ======================================================
def fetch_live_cpi():
    try:
        # Placeholder for real API
        return DEFAULT_CPI, "Default (API Placeholder)"
    except:
        return None, None


def load_uploaded_cpi(file):
    if file is None:
        return None, None
    try:
        cpi_df = pd.read_excel(file)
        return dict(zip(cpi_df["Year"], cpi_df["CPI"])), "Uploaded File"
    except:
        return None, None


def get_hybrid_cpi(uploaded_cpi_file=None, use_api=True):
    if use_api:
        api_cpi, source = fetch_live_cpi()
        if api_cpi:
            return api_cpi, source

    uploaded_cpi, source = load_uploaded_cpi(uploaded_cpi_file)
    if uploaded_cpi:
        return uploaded_cpi, source

    return DEFAULT_CPI, "Built-in Default"


# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Claim Analytics", "Anomaly Detection", "Forecasting", "About"]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", ["csv", "xlsx"])

st.sidebar.subheader("CPI Inflation Adjustment")
use_api = st.sidebar.checkbox("Use Live CPI API", True)
uploaded_cpi_file = st.sidebar.file_uploader("Upload CPI Table (Year, CPI)", ["xlsx"])


# ======================================================
# DATA LOADER
# ======================================================
def load_data(file):
    if file is None:
        return None
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df["Excel Row"] = df.index + 2
    return df


df = load_data(uploaded_file)


# ======================================================
# PREPROCESS
# ======================================================
def preprocess(df):
    df = df.copy()

    if "DOB" in df.columns:
        df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
        df["Age"] = pd.Timestamp.today().year - df["DOB"].dt.year

    if "Visit Date" in df.columns:
        df["Visit Date"] = pd.to_datetime(df["Visit Date"], errors="coerce")

    numeric_cols = [
        "Total Bill (RM)",
        "No. of MC Days",
        "Insurance Amount (RM)",
        "Patient Excess Amount (RM)"
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


if df is not None:
    df = preprocess(df)

    # ---- Apply Hybrid CPI ----
    if "Visit Date" in df.columns and "Total Bill (RM)" in df.columns:
        cpi_table, cpi_source = get_hybrid_cpi(uploaded_cpi_file, use_api)

        current_year = pd.Timestamp.today().year
        current_cpi = cpi_table.get(current_year, max(cpi_table.values()))

        df["Claim Year"] = df["Visit Date"].dt.year
        df["CPI"] = df["Claim Year"].map(cpi_table).fillna(current_cpi)

        df["Adjusted Total Bill (RM)"] = (
            df["Total Bill (RM)"] * (current_cpi / df["CPI"])
        )


# ======================================================
# HOME
# ======================================================
if page == "Home":
    st.title("Medical Claims AI Dashboard")

    if df is not None:
        st.success("Dataset loaded successfully")

        st.info(f"💹 CPI Source Used: {cpi_source}")

        st.subheader("Original vs Inflation Adjusted Claims")

        compare_df = df[["Excel Row", "Visit Date", "Total Bill (RM)", "Adjusted Total Bill (RM)"]].dropna().head(50)
        st.dataframe(compare_df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.plot(compare_df["Total Bill (RM)"].values, label="Original Bill")
        ax.plot(compare_df["Adjusted Total Bill (RM)"].values, linestyle="--", label="Adjusted Bill")
        ax.set_title("Original vs CPI Adjusted Claim Amount")
        ax.set_ylabel("RM")
        ax.legend()
        st.pyplot(fig)

    else:
        st.info("Upload a dataset to begin.")


# ======================================================
# CLAIM ANALYTICS
# ======================================================
elif page == "Claim Analytics":
    st.title("Claim Distribution Analytics")

    if df is None:
        st.warning("Upload dataset first.")
        st.stop()

    cost_col = "Adjusted Total Bill (RM)"

    col1, col2 = st.columns(2)

    if "Clinic Code" in df.columns:
        clinic_stats = df.groupby("Clinic Code")[cost_col].agg(["count", "sum"]).sort_values("sum", ascending=False)
        col1.subheader("Top Clinics by Adjusted Cost")
        col1.dataframe(clinic_stats.head(10))
        col1.bar_chart(clinic_stats["sum"].head(10))

    if "Clinic State" in df.columns:
        state_stats = df.groupby("Clinic State")[cost_col].sum().sort_values(ascending=False)
        col2.subheader("Cost by Clinic State")
        col2.dataframe(state_stats)
        col2.bar_chart(state_stats)

    if "Diagnosis 1" in df.columns:
        st.subheader("Diagnosis Cost Impact (Adjusted)")
        diag_stats = df.groupby("Diagnosis 1")[cost_col].sum().sort_values(ascending=False)
        st.dataframe(diag_stats.head(20))
        st.bar_chart(diag_stats.head(20))


# ======================================================
# ANOMALY DETECTION
# ======================================================
elif page == "Anomaly Detection":
    st.title("Abnormal / Abusive Claim Detection")

    data = df.dropna(subset=["Adjusted Total Bill (RM)", "Age"]).copy()

    feature_cols = ["Adjusted Total Bill (RM)", "Age"]

    if "No. of MC Days" in data.columns:
        feature_cols.append("No. of MC Days")

    X = data[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = st.slider("Expected Abnormal Rate (%)", 1, 20, 5) / 100

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_scaled)

    data["Anomaly Score"] = -iso.decision_function(X_scaled)
    data["Anomaly"] = iso.predict(X_scaled)

    anomalies = data[data["Anomaly"] == -1]

    st.metric("Detected Abnormal Claims", len(anomalies))

    show_cols = ["Excel Row"] + feature_cols + ["Anomaly Score"]
    st.dataframe(anomalies[show_cols].sort_values("Anomaly Score", ascending=False).head(30), use_container_width=True)


# ======================================================
# FORECASTING
# ======================================================
elif page == "Forecasting":
    st.title("Monthly Claim Cost Forecasting")

    ts = df.dropna(subset=["Visit Date", "Adjusted Total Bill (RM)"])

    monthly = ts.set_index("Visit Date").resample("M")["Adjusted Total Bill (RM)"].sum()

    horizon = st.slider("Forecast Months", 3, 24, 6)

    model = SARIMAX(monthly, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)

    res = model.fit(disp=False)
    fc = res.get_forecast(steps=horizon).summary_frame()

    fig, ax = plt.subplots()
    monthly.plot(ax=ax, label="History")
    fc["mean"].plot(ax=ax, linestyle="--", label="Forecast")
    ax.fill_between(fc.index, fc["mean_ci_lower"], fc["mean_ci_upper"], alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(fc[["mean", "mean_ci_lower", "mean_ci_upper"]])


# ======================================================
# ABOUT
# ======================================================
elif page == "About":
    st.title("About")
    st.write("""
    • Hybrid CPI inflation engine  
    • Old vs adjusted claim comparison  
    • IsolationForest anomaly detection  
    • SARIMA forecasting on adjusted costs  
    • Excel row audit tracking  
    """)
