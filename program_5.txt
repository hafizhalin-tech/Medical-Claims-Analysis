import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from collections import Counter

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
        "EDA",
        "Classification (Case Status)",
        "Cost Prediction",
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
    st.write("AI-powered medical insurance analytics using RandomForest & Time-Series models.")

    if df is not None:
        st.success("Dataset loaded successfully")
        st.dataframe(df.head())
    else:
        st.info("Upload a dataset to begin.")

# ======================================================
# EDA
# ======================================================
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    if df is None:
        st.warning("Upload dataset first.")
        st.stop()

    st.subheader("Summary")
    st.write(df.describe(include="all"))

    if "Diagnosis 1" in df.columns:
        st.subheader("Top Diagnoses")
        st.bar_chart(df["Diagnosis 1"].value_counts().head(15))

    if "Total Bill (RM)" in df.columns:
        st.subheader("Total Bill Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Total Bill (RM)"].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    if {"Age", "Total Bill (RM)", "Diagnosis 1"}.issubset(df.columns):
        st.subheader("Correlation: Age – Diagnosis – Cost")
        enc = LabelEncoder()
        temp = df.dropna(subset=["Age", "Total Bill (RM)", "Diagnosis 1"])
        temp["Diagnosis Encoded"] = enc.fit_transform(temp["Diagnosis 1"].astype(str))
        corr = temp[["Age", "Diagnosis Encoded", "Total Bill (RM)"]].corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ======================================================
# CLASSIFICATION
# ======================================================
elif page == "Classification (Case Status)":
    st.title("Case Status Prediction (RandomForest)")

    if df is None or "Case Status" not in df.columns:
        st.error("Dataset must contain Case Status.")
        st.stop()

    data = df.dropna(subset=["Diagnosis 1", "Case Status", "Age"]).copy()

    enc_diag = LabelEncoder()
    enc_state = LabelEncoder()
    enc_status = LabelEncoder()

    data["Diagnosis Encoded"] = enc_diag.fit_transform(data["Diagnosis 1"].astype(str))
    data["Clinic State Encoded"] = (
        enc_state.fit_transform(data["Clinic State"].astype(str))
        if "Clinic State" in data.columns else 0
    )
    data["Case Status Encoded"] = enc_status.fit_transform(data["Case Status"].astype(str))

    features = ["Diagnosis Encoded", "Age", "Clinic State Encoded"]
    X = data[features].fillna(0)
    y = data["Case Status Encoded"]

    counts = Counter(y)
    can_stratify = len(counts) > 1 and min(counts.values()) >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if can_stratify else None
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.metric("Accuracy", f"{acc*100:.2f}%")

    st.subheader("Feature Importance")
    fi = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    st.bar_chart(fi.set_index("Feature"))

    st.subheader("SHAP-like (Permutation Importance)")
    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    perm_df = pd.DataFrame({
        "Feature": features,
        "Importance": perm.importances_mean
    }).sort_values("Importance", ascending=False)
    st.bar_chart(perm_df.set_index("Feature"))

# ======================================================
# COST PREDICTION
# ======================================================
elif page == "Cost Prediction":
    st.title("Total Bill Prediction")

    if df is None:
        st.stop()

    data = df.dropna(subset=["Diagnosis 1", "Total Bill (RM)", "Age"]).copy()
    enc = LabelEncoder()
    data["Diagnosis Encoded"] = enc.fit_transform(data["Diagnosis 1"].astype(str))

    features = ["Diagnosis Encoded", "Age"]
    if "Clinic State" in data.columns:
        data["Clinic State Encoded"] = LabelEncoder().fit_transform(data["Clinic State"].astype(str))
        features.append("Clinic State Encoded")

    X = data[features]
    y = data["Total Bill (RM)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    st.metric("MAE", f"RM {mean_absolute_error(y_test, preds):,.2f}")
    st.metric("RMSE", f"RM {np.sqrt(mean_squared_error(y_test, preds)):,.2f}")

# ======================================================
# ANOMALY DETECTION
# ======================================================
elif page == "Anomaly Detection":
    st.title("Abnormal / Abusive Claim Detection")

    data = df.dropna(subset=["Total Bill (RM)", "Age"]).copy()
    features = ["Total Bill (RM)", "Age"]
    if "No. of MC Days" in data.columns:
        features.append("No. of MC Days")

    iso = IsolationForest(contamination=0.05, random_state=42)
    data["Anomaly"] = iso.fit_predict(data[features].fillna(0))

    anomalies = data[data["Anomaly"] == -1]
    st.metric("Detected Anomalies", len(anomalies))
    st.dataframe(anomalies.head(20))

# ======================================================
# FORECASTING (REAL)
# ======================================================
elif page == "Forecasting":
    st.title("Monthly Claim Cost Forecasting")

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

    # Optional: SARIMA vs Linear Trend
    st.subheader("Model Comparison")
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
    • RandomForest Classification & Regression  
    • SHAP-like Permutation Importance  
    • Abnormal Claim Detection (IsolationForest)  
    • Proper SARIMA Time-Series Forecasting  
    • Designed for Medical Insurance Analytics
    """)
