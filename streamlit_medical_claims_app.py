import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import shap

st.set_page_config(page_title="Medical Claims AI Dashboard", layout="wide")

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "AI Models", "Cost Prediction", "About"])

st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv", "xlsx"])

def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

df = load_data(uploaded_file)

# -------------------------------- HOME --------------------------------
if page == "Home":
    st.title("🏥 Medical Claims AI Dashboard")
    st.write("This is an AI-powered dashboard for analysing medical claims.")
    st.write("**Diagnosis 1** is used as the main diagnostic variable.")
    if df is not None:
        st.success("Data successfully loaded!")
        st.write(df.head())
    else:
        st.warning("Please upload a dataset using the sidebar.")

# -------------------------------- EDA --------------------------------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        st.subheader("Dataset Overview")
        st.write(df.head())
        st.write(df.describe())

        # Age calculation
        if "DOB" in df.columns:
            df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
            df["Age"] = (pd.Timestamp("today") - df["DOB"]).dt.days // 365

        # Plot diagnosis frequency
        if "Diagnosis 1" in df.columns:
            st.subheader("Diagnosis 1 Frequency")
            fig, ax = plt.subplots()
            df["Diagnosis 1"].value_counts().head(20).plot(kind="bar", ax=ax)
            st.pyplot(fig)

        # Cost distribution
        if "Total Bill (RM)" in df.columns:
            st.subheader("Total Bill Distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(df["Total Bill (RM)"], kde=True, ax=ax2)
            st.pyplot(fig2)

# -------------------------------- AI MODELS --------------------------------
elif page == "AI Models":
    st.title("🤖 Claim Approval Prediction (AI Model)")
    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        if "Case Status" not in df.columns:
            st.error("Missing 'Case Status' column.")
        else:
            # Preprocess
            data = df.copy()
            data = data.dropna(subset=["Diagnosis 1", "Case Status"])  # Clean

            # Encode diagnosis
            le_diag = LabelEncoder()
            data["Diagnosis 1 Encoded"] = le_diag.fit_transform(data["Diagnosis 1"].astype(str))

            # Encode Case Status
            le_status = LabelEncoder()
            data["Case Status Encoded"] = le_status.fit_transform(data["Case Status"].astype(str))

            features = ["Diagnosis 1 Encoded"]
            if "Total Bill (RM)" in df.columns:
                features.append("Total Bill (RM)")

            X = data[features]
            y = data["Case Status Encoded"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = XGBClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            accuracy = accuracy_score(y_test, preds)
            st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

            # SHAP Explainability
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            st.subheader("Feature Importance (SHAP)")
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig3)

# -------------------------------- COST PREDICTION --------------------------------
elif page == "Cost Prediction":
    st.title("💰 Total Bill Prediction (Regression Model)")
    if df is None:
        st.warning("Upload data first.")
    else:
        if "Total Bill (RM)" not in df.columns:
            st.error("Missing 'Total Bill (RM)' column.")
        else:
            data = df.dropna(subset=["Diagnosis 1", "Total Bill (RM)"])

            le = LabelEncoder()
            data["Diagnosis 1 Encoded"] = le.fit_transform(data["Diagnosis 1"].astype(str))

            X = data[["Diagnosis 1 Encoded"]]
            y = data["Total Bill (RM)"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = XGBRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            st.metric("MAE", f"RM {mae:,.2f}")
            st.metric("RMSE", f"RM {rmse:,.2f}")

# -------------------------------- ABOUT --------------------------------
elif page == "About":
    st.title("ℹ️ About This Dashboard")
    st.write("Developed as a unified AI-driven diagnostic and cost analysis tool.")
