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

    # Track original Excel row
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
    st.write("AI-powered abnormal claim detection and cost forecasting.")

    if df is not None:
        st.success("Dataset loaded successfully")
        st.dataframe(df.head())
    else:
        st.info("Upload a dataset to begin.")

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

    # ---- Explain Drivers ----
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

    st.subheader("Top Abnormal Claims with Drivers")

    display_df = (
        anomalies[show_cols]
        .sort_values("Anomaly Score", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )

    

    st.dataframe(display_df.style.apply(highlight_rows, axis=1), use_container_width=True)

    st.subheader("Anomaly Driver Distribution")
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

    if len(monthly) < 6:
        st.warning("Not enough history for forecasting.")
        st.stop()

    horizon = st.slider("Forecast Months", 3, 24, 6)

    model = SARIMAX(
        monthly,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    res = model.fit(disp=False)
    fc = res.get_forecast(steps=horizon).summary_frame()

    fig, ax = plt.subplots()
    monthly.plot(ax=ax, label="History")
    fc["mean"].plot(ax=ax, linestyle="--", label="Forecast")
    ax.fill_between(fc.index, fc["mean_ci_lower"], fc["mean_ci_upper"], alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Forecast Table")
    st.dataframe(fc[["mean", "mean_ci_lower", "mean_ci_upper"]])

    st.subheader("Linear Trend Comparison")
    lr = LinearRegression()
    X_lr = np.arange(len(monthly)).reshape(-1, 1)
    lr.fit(X_lr, monthly.values)
    st.write("Trend Slope:", lr.coef_[0])

# ======================================================
# ABOUT
# ======================================================
elif page == "About":
    st.title("About")
    st.write("""
    • IsolationForest for abnormal claim detection  
    • Excel row tracking for audit trail  
    • Driver-based anomaly explanation  
    • SARIMA forecasting  
    • Visual highlighting of risky claims  
    """)
