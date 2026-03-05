import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config("Medical Claims AI Dashboard", layout="wide")

# ======================================================
# BUILT-IN CPI TABLE (Malaysia CPI Approximation)
# ======================================================
CPI_TABLE = {
    2018: 120.66,
    2019: 121.46,
    2020: 120.08,
    2021: 123.05,
    2022: 127.21,
    2023: 130.38,
    2024: 132.77,
    2025: 136.50
}

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Claim Analytics", "Anomaly Detection", "Forecasting", "About"]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel Dataset", ["csv", "xlsx"])


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

    # Track original Excel row
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


# ======================================================
# APPLY CPI ADJUSTMENT
# ======================================================
if df is not None:

    df = preprocess(df)

    if "Visit Date" in df.columns and "Total Bill (RM)" in df.columns:

        current_year = pd.Timestamp.today().year
        current_cpi = CPI_TABLE.get(current_year, max(CPI_TABLE.values()))

        df["Claim Year"] = df["Visit Date"].dt.year

        df["CPI"] = df["Claim Year"].map(CPI_TABLE).fillna(current_cpi)

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

        st.subheader("Original vs Inflation Adjusted Claims")

        compare_df = df[
            ["Excel Row", "Visit Date", "Total Bill (RM)", "Adjusted Total Bill (RM)"]
        ].dropna().head(50)

        st.dataframe(compare_df, use_container_width=True)

        fig, ax = plt.subplots()

        ax.plot(compare_df["Total Bill (RM)"].values, label="Original Bill")

        ax.plot(
            compare_df["Adjusted Total Bill (RM)"].values,
            linestyle="--",
            label="Adjusted Bill"
        )

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

    # --------------------------------------------------
    # Clinic Cost Analysis
    # --------------------------------------------------
    if "Clinic Code" in df.columns:

        clinic_stats = (
            df.groupby("Clinic Code")[cost_col]
            .agg(["count", "sum"])
            .sort_values("sum", ascending=False)
        )

        col1.subheader("Top Clinics by Adjusted Cost")

        col1.dataframe(clinic_stats.head(10))

        col1.bar_chart(clinic_stats["sum"].head(10))

    # --------------------------------------------------
    # State Cost Analysis
    # --------------------------------------------------
    if "Clinic State" in df.columns:

        state_stats = (
            df.groupby("Clinic State")[cost_col]
            .sum()
            .sort_values(ascending=False)
        )

        col2.subheader("Cost by Clinic State")

        col2.dataframe(state_stats)

        col2.bar_chart(state_stats)

    # --------------------------------------------------
    # Diagnosis Cost
    # --------------------------------------------------
    if "Diagnosis 1" in df.columns:

        st.subheader("Diagnosis Cost Impact (Adjusted)")

        diag_stats = (
            df.groupby("Diagnosis 1")[cost_col]
            .sum()
            .sort_values(ascending=False)
        )

        st.dataframe(diag_stats.head(20))

        st.bar_chart(diag_stats.head(20))

    # --------------------------------------------------
    # CLAIM TYPE DISTRIBUTION
    # --------------------------------------------------
    st.subheader("Claim Type Distribution")

    possible_cols = [
        "Claim Type",
        "Claims Type",
        "Type",
        "Payment Type",
        "Case/ Claims Type"
    ]

    claim_type_col = None

    for col in possible_cols:
        if col in df.columns:
            claim_type_col = col
            break

    if claim_type_col is not None:

        claim_counts = df[claim_type_col].value_counts()

        st.dataframe(claim_counts.rename("Number of Cases"))

        fig2, ax2 = plt.subplots()

        ax2.bar(
            claim_counts.index.astype(str),
            claim_counts.values
        )

        ax2.set_title("Number of Cases by Claim Type")
        ax2.set_ylabel("Number of Cases")
        ax2.set_xlabel("Claim Type")

        st.pyplot(fig2)

    else:
        st.warning("No Claim Type column found in dataset.")


# ======================================================
# ANOMALY DETECTION
# ======================================================
elif page == "Anomaly Detection":

    st.title("Abnormal / Abusive Claim Detection")

    data = df.dropna(
        subset=["Adjusted Total Bill (RM)", "Age"]
    ).copy()

    feature_cols = [
        "Adjusted Total Bill (RM)",
        "Age"
    ]

    if "No. of MC Days" in data.columns:
        feature_cols.append("No. of MC Days")

    X = data[feature_cols].fillna(0)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    contamination = st.slider(
        "Expected Abnormal Rate (%)",
        1,
        20,
        5
    ) / 100

    iso = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    iso.fit(X_scaled)

    data["Anomaly Score"] = -iso.decision_function(X_scaled)

    data["Anomaly"] = iso.predict(X_scaled)

    anomalies = data[data["Anomaly"] == -1]

    st.metric(
        "Detected Abnormal Claims",
        len(anomalies)
    )

    show_cols = ["Excel Row"] + feature_cols + ["Anomaly Score"]

    st.dataframe(
        anomalies[show_cols]
        .sort_values("Anomaly Score", ascending=False)
        .head(30),
        use_container_width=True
    )


# ======================================================
# FORECASTING
# ======================================================
elif page == "Forecasting":

    st.title("Monthly Claim Cost Forecasting")

    ts = df.dropna(
        subset=["Visit Date", "Adjusted Total Bill (RM)"]
    )

    monthly = (
        ts.set_index("Visit Date")
        .resample("M")["Adjusted Total Bill (RM)"]
        .sum()
    )

    horizon = st.slider(
        "Forecast Months",
        3,
        24,
        6
    )

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

    fc["mean"].plot(
        ax=ax,
        linestyle="--",
        label="Forecast"
    )

    ax.fill_between(
        fc.index,
        fc["mean_ci_lower"],
        fc["mean_ci_upper"],
        alpha=0.3
    )

    ax.legend()

    st.pyplot(fig)

    st.dataframe(
        fc[["mean", "mean_ci_lower", "mean_ci_upper"]]
    )


# ======================================================
# ABOUT
# ======================================================
elif page == "About":

    st.title("About")

    st.write("""
• CPI-based inflation adjustment  
• Original vs adjusted claim comparison  
• Claim distribution analytics  
• Isolation Forest anomaly detection  
• SARIMA monthly forecasting  
• Excel row audit tracking
""")
