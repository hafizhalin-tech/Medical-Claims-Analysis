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
        "Anomaly Detection",
        "Forecasting",
        "Clinic Risk Scoring",
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
# FILTERS
# ======================================================
def apply_filters(df):
    st.sidebar.subheader("Filters")

    if "Clinic State" in df.columns:
        states = st.sidebar.multiselect("Clinic State", sorted(df["Clinic State"].dropna().unique()))
    else:
        states = []

    if "Clinic Code" in df.columns:
        clinics = st.sidebar.multiselect("Clinic Code", sorted(df["Clinic Code"].dropna().unique()))
    else:
        clinics = []

    if "Diagnosis 1" in df.columns:
        diags = st.sidebar.multiselect("Diagnosis", sorted(df["Diagnosis 1"].dropna().unique()))
    else:
        diags = []

    if "Visit Date" in df.columns:
        min_d, max_d = df["Visit Date"].min(), df["Visit Date"].max()
        date_range = st.sidebar.date_input("Visit Date Range", [min_d, max_d])
    else:
        date_range = None

    f = df.copy()

    if states:
        f = f[f["Clinic State"].isin(states)]
    if clinics:
        f = f[f["Clinic Code"].isin(clinics)]
    if diags:
        f = f[f["Diagnosis 1"].isin(diags)]
    if date_range and len(date_range) == 2:
        f = f[(f["Visit Date"] >= pd.to_datetime(date_range[0])) &
              (f["Visit Date"] <= pd.to_datetime(date_range[1]))]

    return f


# ======================================================
# HOME
# ======================================================
if page == "Home":
    st.title("Medical Claims AI Dashboard")

    if df is not None:
        fdf = apply_filters(df)
        st.success(f"Loaded {len(fdf)} filtered records")
        st.dataframe(fdf.head())
    else:
        st.info("Upload dataset to begin.")

# ======================================================
# ANOMALY DETECTION
# ======================================================
elif page == "Anomaly Detection":
    st.title("Abnormal / Abusive Claim Detection")

    if df is None:
        st.stop()

    data = apply_filters(df)
    data = data.dropna(subset=["Total Bill (RM)", "Age"]).copy()

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

    st.dataframe(
        anomalies[show_cols]
        .sort_values("Anomaly Score", ascending=False)
        .head(30),
        use_container_width=True
    )

# ======================================================
# CLINIC RISK SCORING
# ======================================================
elif page == "Clinic Risk Scoring":
    st.title("AI Risk Scoring per Clinic")

    if df is None:
        st.stop()

    data = apply_filters(df)
    data = data.dropna(subset=["Total Bill (RM)", "Age"]).copy()

    feature_cols = ["Total Bill (RM)", "Age"]
    if "No. of MC Days" in data.columns:
        feature_cols.append("No. of MC Days")

    X = StandardScaler().fit_transform(data[feature_cols].fillna(0))

    iso = IsolationForest(contamination=0.05, random_state=42)
    data["Anomaly"] = iso.fit_predict(X)

    clinic_stats = data.groupby("Clinic Code").agg(
        Claims=("Clinic Code", "count"),
        AnomalyRate=("Anomaly", lambda x: (x == -1).mean()),
        AvgBill=("Total Bill (RM)", "mean"),
        AvgMC=("No. of MC Days", "mean")
    ).fillna(0)

    clinic_stats["RiskScore"] = (
        clinic_stats["AnomalyRate"] * 0.4 +
        clinic_stats["AvgBill"] / clinic_stats["AvgBill"].max() * 0.4 +
        clinic_stats["AvgMC"] / clinic_stats["AvgMC"].max() * 0.2
    )

    clinic_stats["RiskScore"] = (clinic_stats["RiskScore"] /
                                 clinic_stats["RiskScore"].max()) * 100

    clinic_stats = clinic_stats.sort_values("RiskScore", ascending=False)

    st.subheader("Clinic Risk Ranking")
    st.dataframe(clinic_stats.round(2), use_container_width=True)

    st.subheader("Risk Score Chart")
    st.bar_chart(clinic_stats["RiskScore"])

# ======================================================
# FORECASTING
# ======================================================
elif page == "Forecasting":
    st.title("Monthly Claim Cost Forecasting")

    if df is None:
        st.stop()

    ts = apply_filters(df)
    ts = ts.dropna(subset=["Visit Date", "Total Bill (RM)"])

    monthly = ts.set_index("Visit Date").resample("M")["Total Bill (RM)"].sum()

    horizon = st.slider("Forecast Months", 3, 24, 6)

    model = SARIMAX(monthly, order=(1,1,1), seasonal_order=(1,1,1,12))
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=horizon).summary_frame()

    fig, ax = plt.subplots()
    monthly.plot(ax=ax, label="History")
    fc["mean"].plot(ax=ax, linestyle="--", label="Forecast")
    ax.fill_between(fc.index, fc["mean_ci_lower"], fc["mean_ci_upper"], alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# ======================================================
# ABOUT
# ======================================================
elif page == "About":
    st.title("About")
    st.write("""
    • IsolationForest anomaly detection  
    • AI risk scoring per clinic  
    • Interactive filtering  
    • SARIMA forecasting  
    • Audit-focused analytics  
    """)
