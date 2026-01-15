import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from collections import Counter

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Medical Claims AI Dashboard",
    layout="wide"
)

# ================= SIDEBAR =================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "EDA",
        "Classification (Case Status)",
        "Cost Prediction",
        "Abnormal Claims Detection",
        "Forecasting",
        "About"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

# ================= DATA LOADER =================
def load_data(file):
    if file is None:
        return None
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

df = load_data(uploaded_file)

# ================= COMMON PREPROCESSING =================
def preprocess_common(df):
    df = df.copy()

    # Convert dates
    if "DOB" in df.columns:
        df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
        df["Age"] = pd.Timestamp.today().year - df["DOB"].dt.year

    if "Visit Date" in df.columns:
        df["Visit Date"] = pd.to_datetime(df["Visit Date"], errors="coerce")

    # Numeric conversion
    numeric_cols = [
        "Total Bill (RM)",
        "Insurance Amount (RM)",
        "Patient Excess Amount (RM)",
        "Annual Limit (RM)"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

if df is not None:
    df = preprocess_common(df)

# ================= HOME =================
if page == "Home":
    st.title("🏥 Medical Claims AI Dashboard")

    st.write("""
    **AI-powered medical insurance analytics**
    - RandomForest models
    - Cost prediction
    - Fraud & abuse detection
    - Explainable AI (no SHAP dependency)
    """)

    if df is not None:
        st.success("Dataset loaded successfully")
        st.write(df.head())
    else:
        st.info("Please upload a dataset")

# ================= EDA =================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    if df is None:
        st.warning("Upload a dataset first")
        st.stop()

    st.subheader("Dataset Overview")
    st.write(df.describe(include="all"))

    if "Diagnosis 1" in df.columns:
        st.subheader("Top Diagnoses")
        fig, ax = plt.subplots()
        df["Diagnosis 1"].astype(str).value_counts().head(15).plot.bar(ax=ax)
        st.pyplot(fig)

    if "Total Bill (RM)" in df.columns:
        st.subheader("Total Bill Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Total Bill (RM)"].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    # Correlation
    required = ["Age", "Total Bill (RM)", "Diagnosis 1"]
    if all(col in df.columns for col in required):
        st.subheader("Correlation Analysis")

        le = LabelEncoder()
        temp = df[required].dropna()
        temp["Diagnosis Encoded"] = le.fit_transform(temp["Diagnosis 1"].astype(str))

        corr = temp[["Age", "Diagnosis Encoded", "Total Bill (RM)"]].corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ================= CLASSIFICATION =================
elif page == "Classification (Case Status)":
    st.title("✅ Claim Approval Classification")

    if df is None:
        st.warning("Upload a dataset first")
        st.stop()

    if "Case Status" not in df.columns:
        st.error("Missing 'Case Status'")
        st.stop()

    features = []
    for col in ["Age", "Total Bill (RM)", "Insurance Amount (RM)"]:
        if col in df.columns:
            features.append(col)

    if "Diagnosis 1" in df.columns:
        le_diag = LabelEncoder()
        df["Diagnosis Encoded"] = le_diag.fit_transform(df["Diagnosis 1"].astype(str))
        features.append("Diagnosis Encoded")

    if "Clinic State" in df.columns:
        le_state = LabelEncoder()
        df["Clinic State Encoded"] = le_state.fit_transform(df["Clinic State"].astype(str))
        features.append("Clinic State Encoded")

    data = df.dropna(subset=features + ["Case Status"]).copy()

    le_status = LabelEncoder()
    y = le_status.fit_transform(data["Case Status"].astype(str))
    X = data[features]

    st.write("📊 Case Status Distribution")
    st.write(pd.Series(y).value_counts())

    # ---- SAFE STRATIFICATION FIX ----
    class_counts = Counter(y)
    can_stratify = len(class_counts) > 1 and min(class_counts.values()) >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if can_stratify else None
    )

    if not can_stratify:
        st.warning("Stratified split disabled due to rare class")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    st.metric("Accuracy", f"{acc*100:.2f}%")
    st.text(classification_report(y_test, preds))

    # Feature importance
    st.subheader("Feature Importance")
    fi = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=fi, ax=ax)
    st.pyplot(fig)

    # SHAP-like explanation
    st.subheader("SHAP-like Explanation")
    expl = X_test.copy()
    expl["Prediction"] = preds
    st.write(expl.groupby("Prediction").mean())

# ================= COST PREDICTION =================
elif page == "Cost Prediction":
    st.title("💰 Total Bill Prediction")

    if df is None or "Total Bill (RM)" not in df.columns:
        st.warning("Missing Total Bill data")
        st.stop()

    features = []
    for col in ["Age", "Insurance Amount (RM)"]:
        if col in df.columns:
            features.append(col)

    if "Diagnosis 1" in df.columns:
        le = LabelEncoder()
        df["Diagnosis Encoded"] = le.fit_transform(df["Diagnosis 1"].astype(str))
        features.append("Diagnosis Encoded")

    data = df.dropna(subset=features + ["Total Bill (RM)"])

    X = data[features]
    y = data["Total Bill (RM)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    st.metric("MAE (RM)", f"{mae:,.2f}")
    st.metric("RMSE (RM)", f"{rmse:,.2f}")

    fi = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=fi, ax=ax)
    st.pyplot(fig)

# ================= ABNORMAL CLAIMS =================
elif page == "Abnormal Claims Detection":
    st.title("🚨 Abnormal / Abusive Claims Detection")

    if df is None or "Total Bill (RM)" not in df.columns:
        st.warning("Upload valid dataset")
        st.stop()

    features = ["Total Bill (RM)"]
    if "Age" in df.columns:
        features.append("Age")

    data = df[features].dropna()

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )
    data["Anomaly"] = iso.fit_predict(data)

    df_result = data[data["Anomaly"] == -1]

    st.write("⚠️ Flagged Abnormal Claims")
    st.write(df_result.head(20))

# ================= FORECASTING =================
elif page == "Forecasting":
    st.title("📈 Monthly Claim Cost Forecast")

    if df is None or "Visit Date" not in df.columns:
        st.warning("Missing Visit Date")
        st.stop()

    df_time = df.dropna(subset=["Visit Date", "Total Bill (RM)"]).copy()
    df_time["Month"] = df_time["Visit Date"].dt.to_period("M").astype(str)

    monthly = df_time.groupby("Month")["Total Bill (RM)"].sum().reset_index()

    fig, ax = plt.subplots()
    ax.plot(monthly["Month"], monthly["Total Bill (RM)"], marker="o")
    ax.set_xticklabels(monthly["Month"], rotation=45)
    st.pyplot(fig)

# ================= ABOUT =================
elif page == "About":
    st.title("ℹ️ About")
    st.write("""
    **Medical Claims AI Dashboard**
    - RandomForest-based analytics
    - Explainable AI (no SHAP dependency)
    - Fraud & abuse detection
    - Cost forecasting
    - Production-safe data handling
    """)
