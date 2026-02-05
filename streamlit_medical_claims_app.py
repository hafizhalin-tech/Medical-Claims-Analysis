import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config("Medical Claims AI Dashboard", layout="wide")

# ======================================================
# Sidebar
# ======================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Claim Analytics",
        "Anomaly Detection",
        "Forecasting",
        "About"
    ]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", ["csv", "xlsx"])

# ======================================================
# Data Loader
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
# Preprocessing
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

# ======================================================
# HOME
# ======================================================
if page == "Home":
    st.title("Medical Claims AI Dashboard")

    if df is not None:
        st.success("Dataset loaded successfully")
        st.dataframe(df.head())
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

    col1, col2 = st.columns(2)

    # ---- Clinic Code ----
    if "Clinic Code" in df.columns:
        clinic_stats = df.groupby("Clinic Code")["Total Bill (RM)"].agg(["count", "sum"]).sort_values("sum", ascending=False)

        col1.subheader("Top Clinics by Cost")
        col1.dataframe(clinic_stats.head(10))

        col1.bar_chart(clinic_stats["sum"].head(10))

    # ---- Clinic State ----
    if "Clinic State" in df.columns:
        state_stats = df.groupby("Clinic State")["Total Bill (RM)"].sum().sort_values(ascending=False)

        col2.subheader("Cost by Clinic State")
        col2.dataframe(state_stats)
        col2.bar_chart(state_stats)

    st.divider()

    # ---- Case / Claims Type ----
    if "Case/ Claims Type" in df.columns:
        st.subheader("Case / Claims Type Analysis")

        case_stats = df.groupby("Case/ Claims Type")["Total Bill (RM)"].agg(["count", "sum"]).sort_values("sum", ascending=False)

        st.dataframe(case_stats)
        st.bar_chart(case_stats["sum"])

    st.divider()

    # ---- Diagnosis ----
    if "Diagnosis 1" in df.columns:
        st.subheader("Diagnosis (Disease) Analysis")

        diag_stats = df.groupby("Diagnosis 1")["Total Bill (RM)"].agg(["count", "sum"]).sort_values("sum", ascending=False)

        st.dataframe(diag_stats.head(20))
        st.bar_chart(diag_stats["sum"].head(20))

# ======================================================
# ANOMALY DETECTION
# ======================================================
elif page == "Anomaly Detection":
    st.title("Abnormal / Abusive Claim Detection")

    if df is None:
        st.warning("Upload dataset first.")
        st.stop()

    data = df.dropna(subset=["Total Bill (RM)", "Age"]).copy()

    feature_cols = ["Total Bill (RM)", "Age"]

    if "No. of MC Days" in data.columns:
        feature_cols.append("No. of MC Days")

    if "Insurance Amount (RM)" in data.columns:
        feature_cols.append("Insurance Amount (RM)")

    X = data[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = st.slider("Expected Abnormal Rate (%)", 1, 20, 5) / 100

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_scaled)

    data["Anomaly Score"] = -iso.decision_function(X_scaled)
    data["Anomaly"] = iso.predict(X_scaled)

    z_scores = pd.DataFrame(
        np.abs(X_scaled),
        columns=[f"{c} Deviation" for c in feature_cols],
        index=data.index
    )

    data = pd.concat([data, z_scores], axis=1)

    def top_driver(row):
        devs = row[[f"{c} Deviation" for c in feature_cols]]
        return devs.idxmax().replace(" Deviation", "")

    data["Top Anomaly Driver"] = data.apply(top_driver, axis=1)

    anomalies = data[data["Anomaly"] == -1]

    st.metric("Detected Abnormal Claims", len(anomalies))

    show_cols = ["Excel Row"] + feature_cols + ["Anomaly Score", "Top Anomaly Driver"]

    display_df = anomalies[show_cols].sort_values("Anomaly Score", ascending=False).head(30).reset_index(drop=True)

    def highlight_rows(row):
        return ["background-color: #ffcccc"] * len(row)

    st.dataframe(display_df.style.apply(highlight_rows, axis=1), use_container_width=True)

    st.bar_chart(anomalies["Top Anomaly Driver"].value_counts())

# ======================================================
# FORECASTING
# ======================================================
elif page == "Forecasting":
    st.title("Monthly Claim Cost Forecasting")

    if df is None:
        st.warning("Upload dataset first.")
        st.stop()

    ts = df.dropna(subset=["Visit Date", "Total Bill (RM)"])

    monthly = ts.set_index("Visit Date").resample("M")["Total Bill (RM)"].sum()

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
    • IsolationForest anomaly detection  
    • Clinic & diagnosis analytics  
    • SARIMA forecasting  
    • Excel row audit tracking  
    """)

