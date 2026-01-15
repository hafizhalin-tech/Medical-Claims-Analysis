# medical_claims_dashboard.py
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

# Page layout
st.set_page_config(page_title="Medical Claims AI Dashboard", layout="wide")
sns.set_style("whitegrid")

# ----------------- Helpers -----------------
@st.cache_data
def load_data(file):
    if file is None:
        return None
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

@st.cache_data
def preprocess(df):
    df = df.copy()
    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    # DOB -> Age
    if "DOB" in df.columns:
        df["DOB_parsed"] = pd.to_datetime(df["DOB"], errors="coerce")
        today = pd.to_datetime("today")
        df["Age"] = (today - df["DOB_parsed"]).dt.days // 365
    else:
        df["Age"] = np.nan

    # Visit Date, YearMonth
    if "Visit Date" in df.columns:
        df["Visit Date"] = pd.to_datetime(df["Visit Date"], errors="coerce")
        df["YearMonth"] = df["Visit Date"].dt.to_period("M").astype(str)
    else:
        df["YearMonth"] = np.nan

    # Numeric coercion
    if "Total Bill (RM)" in df.columns:
        df["Total Bill (RM)"] = pd.to_numeric(df["Total Bill (RM)"], errors="coerce")
    if "No. of MC Days" in df.columns:
        df["No. of MC Days"] = pd.to_numeric(df["No. of MC Days"], errors="coerce")
    if "Insurance Amount (RM)" in df.columns:
        df["Insurance Amount (RM)"] = pd.to_numeric(df["Insurance Amount (RM)"], errors="coerce")
    if "Annual Limit (RM)" in df.columns:
        df["Annual Limit (RM)"] = pd.to_numeric(df["Annual Limit (RM)"], errors="coerce")

    # Ensure Diagnosis 1 column exists
    if "Diagnosis 1" not in df.columns:
        df["Diagnosis 1"] = np.nan

    # Clinic State default
    if "Clinic State" not in df.columns:
        df["Clinic State"] = np.nan

    return df

def safe_label_encode(series):
    s = series.fillna("missing").astype(str)
    le = LabelEncoder()
    return le.fit_transform(s), le

def show_feature_effects(df_local, feature, target_col, problem='regression'):
    """
    Simple feature effect visualization:
    - if categorical: mean target per category
    - if numeric: bin into deciles and show mean target per bin
    """
    if feature not in df_local.columns:
        st.info(f"Feature {feature} not present.")
        return

    df_f = df_local[[feature, target_col]].dropna()
    if df_f.empty:
        st.info("No data to plot feature effects.")
        return

    if df_f[feature].dtype == object or df_f[feature].nunique() < 12:
        grp = df_f.groupby(feature)[target_col].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        grp.plot(kind="bar", ax=ax)
        ax.set_ylabel(f"Mean {target_col}")
        ax.set_xlabel(feature)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        # numeric: bin
        df_f["bin"] = pd.qcut(df_f[feature], q=10, duplicates="drop")
        grp = df_f.groupby("bin")[target_col].mean()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        grp.plot(kind="bar", ax=ax)
        ax.set_ylabel(f"Mean {target_col}")
        ax.set_xlabel(feature + " (binned)")
        plt.tight_layout()
        st.pyplot(fig)

# ----------------- UI / Pages -----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page", ["Home", "EDA", "Classification (Case Status)", "Regression (Total Bill)", "Feature Importance", "Anomaly & Forecast", "About"])

uploaded = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
raw = load_data(uploaded)
df = preprocess(raw) if raw is not None else None

# ---------------- Home ----------------
if page == "Home":
    st.title("Medical Claims AI Dashboard")
    st.markdown("This app analyses medical claims data and runs RandomForest-based AI models.\n\n**Only `Diagnosis 1` is used for diagnosis information.**")
    st.markdown("Upload your dataset (CSV or Excel) using the sidebar. The app will try to auto-detect necessary columns.")
    if df is None:
        st.info("No data loaded yet.")
    else:
        st.success("Data loaded and preprocessed.")
        st.dataframe(df.head(200))

# ---------------- EDA ----------------
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    if df is None:
        st.warning("Upload data first.")
    else:
        st.subheader("Dataset overview")
        cols = st.multiselect("Show columns", df.columns.tolist(), default=["Relationship","DOB","Case Status","Visit Date","Clinic State","Total Bill (RM)","Diagnosis 1","Age"])
        st.dataframe(df[cols].head(200))

        st.subheader("Summary statistics")
        st.write(df.describe(include='all'))

        st.subheader("Diagnosis 1 frequency (top 20)")
        topd = df["Diagnosis 1"].fillna("missing").value_counts().head(20)
        fig, ax = plt.subplots(figsize=(8,3))
        topd.plot(kind="bar", ax=ax)
        ax.set_ylabel("Counts")
        st.pyplot(fig)

        if "Total Bill (RM)" in df.columns:
            st.subheader("Total Bill distribution")
            fig, ax = plt.subplots(figsize=(8,3))
            sns.histplot(df["Total Bill (RM)"].dropna(), kde=True, ax=ax, bins=40)
            ax.set_xlabel("Total Bill (RM)")
            st.pyplot(fig)

        # Correlation between Age, Diagnosis 1 (encoded), and Total Bill
        if "Age" in df.columns and "Total Bill (RM)" in df.columns and "Diagnosis 1" in df.columns:
            st.subheader("Correlation: Age / Diagnosis 1 / Total Bill")
            corr_df = df[["Age", "Diagnosis 1", "Total Bill (RM)"]].dropna()
            if not corr_df.empty:
                corr_df["Diagnosis 1 Encoded"] = LabelEncoder().fit_transform(corr_df["Diagnosis 1"].astype(str))
                corr_mat = corr_df[["Age","Diagnosis 1 Encoded","Total Bill (RM)"]].corr()
                st.write(corr_mat)
                fig, ax = plt.subplots(figsize=(4,3))
                sns.heatmap(corr_mat, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.info("Not enough data for correlation (missing Age, Diagnosis 1 or Total Bill).")

# ---------------- Classification ----------------
elif page == "Classification (Case Status)":
    st.title("Predict Case Status (RandomForestClassifier)")
    st.markdown("Target: `Case Status` (e.g., Approved / Rejected). The model uses Diagnosis 1, Age, Clinic State, and Total Bill if available.")

    if df is None:
        st.warning("Upload data first.")
    elif "Case Status" not in df.columns:
        st.error("Column `Case Status` not found in dataset.")
    else:
        data = df.dropna(subset=["Diagnosis 1", "Case Status"]).copy()

        # features and encoding
        data["Diag1_enc"] = LabelEncoder().fit_transform(data["Diagnosis 1"].astype(str))
        data["ClinicState_enc"] = LabelEncoder().fit_transform(data["Clinic State"].fillna("missing").astype(str))
        if "Total Bill (RM)" in data.columns:
            data["Total Bill (RM)"] = pd.to_numeric(data["Total Bill (RM)"], errors="coerce")

        # Ensure Age present (may be NaN)
        if "Age" not in data.columns:
            data["Age"] = np.nan

        feature_cols = ["Diag1_enc", "Age", "ClinicState_enc"]
        if "Total Bill (RM)" in data.columns:
            feature_cols.append("Total Bill (RM)")

        X = data[feature_cols].fillna(0)
        y = LabelEncoder().fit_transform(data["Case Status"].astype(str))

        # Train/test split
        test_size = st.sidebar.slider("Test set proportion (classification)", 5, 50, 20) / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y))>1 else None)

        # Train model
        n_estimators = st.sidebar.slider("n_estimators (RandomForest)", 50, 500, 100)
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc*100:.2f}%")
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()

        # Built-in feature importance
        st.subheader("Built-in Feature Importances")
        fi = pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
        fig, ax = plt.subplots(figsize=(6,3))
        sns.barplot(x="importance", y="feature", data=fi, ax=ax)
        st.pyplot(fig)
        st.write(fi)

        # Permutation importance (SHAP-like)
        st.subheader("Permutation Importances (approximate SHAP-like)")
        try:
            perm = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
            perm_df = pd.DataFrame({"feature": feature_cols, "perm_importance": perm.importances_mean}).sort_values("perm_importance", ascending=False)
            fig, ax = plt.subplots(figsize=(6,3))
            sns.barplot(x="perm_importance", y="feature", data=perm_df, ax=ax)
            st.pyplot(fig)
            st.write(perm_df)
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")

        # Feature effect plots (top features)
        st.subheader("Feature effect plots (approximate)")
        top_feats = fi["feature"].tolist()[:3]
        for f in top_feats:
            show_feature_effects(data.assign(**{c: data[c] for c in feature_cols}), f, "Case Status", problem="classification")

        # Download predictions
        st.subheader("Download predictions (classification)")
        out_df = X_test.copy()
        out_df["predicted_label_num"] = y_pred
        # convert predicted numeric label back if needed (we used LabelEncoder on y)
        # Recreate label encoder mapping:
        label_le = LabelEncoder().fit(data["Case Status"].astype(str))
        try:
            out_df["predicted_label"] = label_le.inverse_transform(out_df["predicted_label_num"].astype(int))
        except Exception:
            out_df["predicted_label"] = out_df["predicted_label_num"]
        st.download_button("Download classification predictions CSV", out_df.to_csv(index=False).encode("utf-8"), "predictions_classification.csv", "text/csv")
        st.write("Sample predictions:")
        st.dataframe(out_df.head(50))

# ---------------- Regression ----------------
elif page == "Regression (Total Bill)":
    st.title("Predict Total Bill (RandomForestRegressor)")
    st.markdown("Target: `Total Bill (RM)` as numeric. Uses Diagnosis 1, Age, Clinic State.")

    if df is None:
        st.warning("Upload data first.")
    elif "Total Bill (RM)" not in df.columns:
        st.error("Column `Total Bill (RM)` not found in dataset.")
    else:
        data = df.dropna(subset=["Diagnosis 1"]).copy()
        data["Total Bill (RM)"] = pd.to_numeric(data["Total Bill (RM)"], errors="coerce")
        data = data.dropna(subset=["Total Bill (RM)"]).copy()

        data["Diag1_enc"] = LabelEncoder().fit_transform(data["Diagnosis 1"].astype(str))
        data["ClinicState_enc"] = LabelEncoder().fit_transform(data["Clinic State"].fillna("missing").astype(str))
        if "Age" not in data.columns:
            data["Age"] = np.nan

        feature_cols = ["Diag1_enc", "Age", "ClinicState_enc"]
        X = data[feature_cols].fillna(0)
        y = data["Total Bill (RM)"]

        test_size = st.sidebar.slider("Test set proportion (regression)", 5, 50, 20) / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        n_estimators = st.sidebar.slider("n_estimators (RandomForest reg)", 50, 500, 100)
        reg = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        try:
            reg.fit(X_train, y_train)
            preds = reg.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            st.metric("MAE", f"RM {mae:,.2f}")
            st.metric("RMSE", f"RM {rmse:,.2f}")
        except Exception as e:
            st.error(f"Regression training failed: {e}")
            st.stop()

        # Built-in importance
        st.subheader("Built-in Feature Importances (regression)")
        fi = pd.DataFrame({"feature": feature_cols, "importance": reg.feature_importances_}).sort_values("importance", ascending=False)
        fig, ax = plt.subplots(figsize=(6,3))
        sns.barplot(x="importance", y="feature", data=fi, ax=ax)
        st.pyplot(fig)
        st.write(fi)

        # Permutation importances
        st.subheader("Permutation Importances (regression)")
        try:
            perm = permutation_importance(reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
            perm_df = pd.DataFrame({"feature": feature_cols, "perm_importance": perm.importances_mean}).sort_values("perm_importance", ascending=False)
            fig, ax = plt.subplots(figsize=(6,3))
            sns.barplot(x="perm_importance", y="feature", data=perm_df, ax=ax)
            st.pyplot(fig)
            st.write(perm_df)
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")

        # Feature effects
        st.subheader("Feature effect plots (approximate)")
        top_feats = fi["feature"].tolist()[:3]
        for f in top_feats:
            show_feature_effects(data.assign(**{c: data[c] for c in feature_cols}), f, "Total Bill (RM)", problem="regression")

        # Download predictions
        st.subheader("Download regression predictions")
        out = X_test.copy()
        out["predicted_cost"] = preds
        st.download_button("Download regression predictions CSV", out.to_csv(index=False).encode("utf-8"), "predictions_regression.csv", "text/csv")
        st.write("Sample predictions:")
        st.dataframe(out.head(50))

# ---------------- Feature Importance Page ----------------
elif page == "Feature Importance":
    st.title("Feature Importance Comparison Dashboard")
    st.markdown("Compare built-in RandomForest importances vs permutation importances (classification & regression)")

    if df is None:
        st.warning("Upload data first.")
    else:
        st.write("This page requires you to run either the Classification or Regression pages first. Feature importances shown on those pages are computed from trained models and printed there.")
        st.info("Go to 'Classification (Case Status)' or 'Regression (Total Bill)' to compute model importances and permutation importances.")

# ---------------- Anomaly & Forecast ----------------
elif page == "Anomaly & Forecast":
    st.title("Anomaly Detection & Monthly Forecast")
    if df is None:
        st.warning("Upload data first.")
    else:
        adf = df.copy()
        # Prepare features
        if "Total Bill (RM)" in adf.columns:
            adf["Total Bill (RM)"] = pd.to_numeric(adf["Total Bill (RM)"], errors="coerce")
        if "No. of MC Days" in adf.columns:
            adf["No. of MC Days"] = pd.to_numeric(adf["No. of MC Days"], errors="coerce")

        adf["Diag1_enc"] = LabelEncoder().fit_transform(adf["Diagnosis 1"].astype(str))
        adf["ClinicState_enc"] = LabelEncoder().fit_transform(adf["Clinic State"].fillna("missing").astype(str)) if "Clinic State" in adf.columns else 0
        if "Age" not in adf.columns:
            adf["Age"] = np.nan

        features_anom = ["Total Bill (RM)", "Diag1_enc", "ClinicState_enc"]
        if "No. of MC Days" in adf.columns:
            features_anom.append("No. of MC Days")
        if "Age" in adf.columns:
            features_anom.append("Age")

        adf_anom = adf[features_anom].fillna(0)

        st.subheader("Anomaly Detection (IsolationForest)")
        contamination = st.slider("Anomaly contamination proportion", 0.001, 0.1, 0.01)
        iso = IsolationForest(contamination=contamination, random_state=42)
        try:
            iso.fit(adf_anom)
            scores = iso.decision_function(adf_anom)
            preds = iso.predict(adf_anom)
            adf["anomaly_score"] = scores
            adf["anomaly_flag"] = preds == -1
            st.write("Anomalies detected:", int(adf["anomaly_flag"].sum()))
            st.dataframe(adf[adf["anomaly_flag"]].sort_values("anomaly_score").head(200))
            st.download_button("Download anomalies CSV", adf[adf["anomaly_flag"]].to_csv(index=False).encode("utf-8"), "anomalies.csv", "text/csv")
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")

        st.markdown("---")
        st.subheader("Monthly Total Bill Forecast (linear trend + moving average)")
        if "Visit Date" not in adf.columns or "Total Bill (RM)" not in adf.columns:
            st.info("Visit Date and Total Bill (RM) required for forecasting.")
        else:
            ts = adf.dropna(subset=["Visit Date"]).copy()
            ts["Visit Date"] = pd.to_datetime(ts["Visit Date"], errors="coerce")
            ts = ts.dropna(subset=["Visit Date"])
            ts["YearMonth"] = ts["Visit Date"].dt.to_period("M").astype(str)
            monthly = ts.groupby("YearMonth")["Total Bill (RM)"].sum().reset_index()
            monthly["Total Bill (RM)"] = pd.to_numeric(monthly["Total Bill (RM)"], errors="coerce").fillna(0)
            monthly = monthly.sort_values("YearMonth").reset_index(drop=True)
            monthly["idx"] = np.arange(len(monthly))

            # Fit linear regression on index
            lr = LinearRegression()
            X_time = monthly[["idx"]]
            y_time = monthly["Total Bill (RM)"]
            if len(X_time) >= 2:
                lr.fit(X_time, y_time)

                n_forecast = st.slider("Forecast months", 1, 24, 6)
                future_idx = np.arange(len(monthly), len(monthly) + n_forecast).reshape(-1,1)
                future_pred = lr.predict(future_idx)

                last_period = pd.Period(monthly["YearMonth"].iloc[-1])
                future_months = [(last_period + i).strftime("%Y-%m") for i in range(1, n_forecast+1)]
                forecast_df = pd.DataFrame({"YearMonth": future_months, "Forecasted Total Bill (RM)": future_pred})

                # Moving average for quick smoothing
                window = st.slider("Moving average window (months)", 1, max(1, len(monthly)//2), min(3, max(1, len(monthly)//2)))
                monthly["ma"] = monthly["Total Bill (RM)"].rolling(window=window, min_periods=1).mean()

                # Plot
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(monthly["YearMonth"], monthly["Total Bill (RM)"], label="Historical")
                ax.plot(monthly["YearMonth"], monthly["ma"], label=f"MA({window})", linestyle="--")
                ax.plot(forecast_df["YearMonth"], forecast_df["Forecasted Total Bill (RM)"], label="Forecast", marker="o")
                # set xticks sparingly
                ticks = list(range(0, len(monthly)+len(forecast_df), max(1, (len(monthly)+len(forecast_df))//10)))
                labels = list(monthly["YearMonth"]) + list(forecast_df["YearMonth"])
                ax.set_xticks(ticks)
                ax.set_xticklabels([labels[i] for i in ticks], rotation=45)
                ax.set_ylabel("Total Bill (RM)")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                st.download_button("Download forecast CSV", forecast_df.to_csv(index=False).encode("utf-8"), "forecast.csv", "text/csv")
            else:
                st.info("Not enough monthly data to fit forecast (need at least 2 months).")

# ---------------- About ----------------
elif page == "About":
    st.title("About")
    st.markdown(
        "This dashboard:\n\n"
        "- Computes Age from DOB automatically.\n"
        "- Uses only Diagnosis 1 for diagnostic features.\n"
        "- Provides EDA, classification (Case Status), regression (Total Bill), "
        "feature importance and permutation importance (SHAP-like), anomaly detection (IsolationForest), and a simple monthly forecast."
    )
    st.markdown("Dependencies: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn")

# End of app
