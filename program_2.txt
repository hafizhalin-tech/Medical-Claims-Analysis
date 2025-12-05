import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="Medical Claims AI Dashboard", layout="wide")

# ================= Sidebar Navigation =================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "AI Models", "Cost Prediction", "About"])

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

# ================= Data Loader =================
def load_data(file):
    if file is None:
        return None
    if file.name.endswith("csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

df = load_data(uploaded_file)

# ================= HOME =================
if page == "Home":
    st.title("Medical Claims AI Dashboard")
    st.write("AI-powered medical claims analytics using Diagnosis 1 only.")

    if df is not None:
        st.success("Data loaded successfully!")
        st.write(df.head())
    else:
        st.info("Upload a dataset from the sidebar.")

# ================= EDA =================
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    if df is None:
        st.warning("Upload a dataset first.")
    else:
        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe())

        if "Diagnosis 1" in df.columns:
            st.subheader("Diagnosis 1 Frequency")
            fig, ax = plt.subplots()
            df["Diagnosis 1"].astype(str).value_counts().head(20).plot(kind="bar", ax=ax)
            st.pyplot(fig)

        if "Total Bill (RM)" in df.columns:
            st.subheader("Total Bill Distribution")
            df["Total Bill (RM)"] = pd.to_numeric(df["Total Bill (RM)"], errors='coerce')
            fig, ax = plt.subplots()
            sns.histplot(df["Total Bill (RM)"].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

# ================= AI MODELS (CLASSIFICATION) =================
elif page == "AI Models":
    st.title("Claim Approval Prediction (RandomForest)")

    if df is None:
        st.warning("Upload a dataset first.")
    elif "Case Status" not in df.columns:
        st.error("Missing 'Case Status' column.")
    else:
        data = df.dropna(subset=["Diagnosis 1", "Case Status"]).copy()

        le_diag = LabelEncoder()
        data["Diagnosis 1 Encoded"] = le_diag.fit_transform(data["Diagnosis 1"].astype(str))

        le_status = LabelEncoder()
        data["Case Status Encoded"] = le_status.fit_transform(data["Case Status"].astype(str))

        feature_cols = ["Diagnosis 1 Encoded"]
        if "Total Bill (RM)" in data.columns:
            data["Total Bill (RM)"] = pd.to_numeric(data["Total Bill (RM)"], errors='coerce')
            feature_cols.append("Total Bill (RM)")

        X = data[feature_cols].fillna(0)
        y = data["Case Status Encoded"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

        st.subheader("Feature Importance")
        fi = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=fi.sort_values(by="Importance", ascending=False), ax=ax)
        st.pyplot(fig)

        st.subheader("Download Predictions")
        output_df = X_test.copy()
        output_df["Predicted Status"] = le_status.inverse_transform(preds)
        csv_data = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv_data, "predictions.csv", "text/csv")

# ================= COST PREDICTION (REGRESSION) =================
elif page == "Cost Prediction":
    st.title("Total Bill Prediction (RandomForest Regressor)")

    if df is None:
        st.warning("Upload dataset first.")
    elif "Total Bill (RM)" not in df.columns:
        st.error("Missing 'Total Bill (RM)' column.")
    else:
        data = df.dropna(subset=["Diagnosis 1", "Total Bill (RM)"]).copy()
        data["Total Bill (RM)"] = pd.to_numeric(data["Total Bill (RM)"], errors='coerce')

        # Drop rows with missing target
        data = data.dropna(subset=["Total Bill (RM)"])

        le = LabelEncoder()
        data["Diagnosis 1 Encoded"] = le.fit_transform(data["Diagnosis 1"].astype(str))

        X = data[["Diagnosis 1 Encoded"]]
        y = data["Total Bill (RM)"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        col1, col2 = st.columns(2)
        col1.metric("MAE", f"RM {mae:,.2f}")
        col2.metric("RMSE", f"RM {rmse:,.2f}")

        st.subheader("Feature Importance")
        fi = pd.DataFrame({"Feature": ["Diagnosis 1 Encoded"], "Importance": model.feature_importances_})
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=fi, ax=ax)
        st.pyplot(fig)

        st.subheader("Download Cost Predictions")
        out = X_test.copy()
        out["Predicted Cost (RM)"] = preds
        csv_out = out.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cost Prediction CSV", csv_out, "cost_predictions.csv", "text/csv")

# ================= ABOUT =================
elif page == "About":
    st.title("About This Dashboard")
    st.write("- Uses RandomForest AI models")
    st.write("- Only Diagnosis 1 is used")
    st.write("- Includes prediction downloads and EDA")
