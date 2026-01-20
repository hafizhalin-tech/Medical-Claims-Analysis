import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        return pd.read_csv(file)
    return pd.read_excel(file)

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
    st.write("AI-powered medical insurance anomaly detection & forecasting.")

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

    data = df.copy()

    feature_cols = ["Total Bill (RM)", "Age"]
    if "No. of MC Days" in data.columns:
        feature_cols.append("No. of MC Days")
    if "Insurance Amount (RM)" in data.columns:
        feature_cols.append("Insurance Amount (RM)")
    if "Patient Excess Amount (RM)" in data.columns:
        feature_cols.append("Patient Excess Amount (RM)")

    data = data.dropna(subset=feature_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[feature_cols])

    contamination = st.slider("Anomaly Sensitivity", 0.01, 0.2, 0.05)

    iso = IsolationForest(contamination=contamination, random_state=42)
    data["Anomaly"] = iso.fit_predict(X_scaled)
    data["Anomaly Score"] = iso.decision_function(X_scaled)

    anomalies = data[data["Anomaly"] == -1]

    st.metric("Detected Abnormal Claims", len(anomalies))

    # ------------------------------------------
    # Explain anomaly drivers
    # ------------------------------------------
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

    st.subheader("Top Abnormal Claims with Drivers")
    show_cols = feature_cols + ["Anomaly Score", "Top Anomaly Driver"]
    st.dataframe(anomalies[show_cols].sort_values("Anomaly Score").head(30))

    st.subheader("Anomaly Driver Distribution")
    st.bar_chart(data.loc[data["Anomaly"] == -1, "Top Anomaly Driver"].value_counts())

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

    st.subheader("Trend Model Insight")
    lr = LinearRegression()
    X = np.arange(len(monthly)).reshape(-1,1)
    lr.fit(X, monthly.values)
    st.write("Linear trend slope:", lr.coef_[0])

# ======================================================
# ABOUT
# ======================================================
elif page == "About":
    st.title("About")
    st.write("""
    • IsolationForest Abnormal Claim Detection  
    • Feature Deviation Explanation  
    • SARIMA Time-Series Forecasting  
    • Designed for Medical Insurance Risk Analytics
    """)
