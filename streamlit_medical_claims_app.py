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

st.set_page_config(page_title="Medical Claims AI Dashboard", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "AI Models", "Cost Prediction", "Anomaly & Forecast", "About"])

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

# ---------------- Data loader & basic preprocessing ----------------
def load_data(file):
    if file is None:
        return None
    if file.name.endswith("csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

@st.cache_data
def preprocess(df):
    df = df.copy()
    # Parse dates
    if 'DOB' in df.columns:
        df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
        today = pd.to_datetime('today')
        df['Age'] = (today - df['DOB']).dt.days // 365
    else:
        df['Age'] = np.nan

    if 'Visit Date' in df.columns:
        df['Visit Date'] = pd.to_datetime(df['Visit Date'], errors='coerce')
        df['YearMonth'] = df['Visit Date'].dt.to_period('M').astype(str)
    else:
        df['YearMonth'] = np.nan

    # Numeric coercion
    if 'Total Bill (RM)' in df.columns:
        df['Total Bill (RM)'] = pd.to_numeric(df['Total Bill (RM)'], errors='coerce')
    if 'No. of MC Days' in df.columns:
        df['No. of MC Days'] = pd.to_numeric(df['No. of MC Days'], errors='coerce')

    # Fill basic missing
    df['Diagnosis 1'] = df.get('Diagnosis 1', pd.Series(np.nan)).astype(object)
    df['Clinic State'] = df.get('Clinic State', pd.Series(np.nan)).astype(object)

    return df

raw_df = load_data(uploaded_file)
df = preprocess(raw_df) if raw_df is not None else None

# ---------------- Utility functions ----------------
def safe_label_encode(series, name=None):
    series = series.fillna('missing').astype(str)
    le = LabelEncoder()
    return le.fit_transform(series), le

def show_feature_effects(df_local, feature, target, problem='regression'):
    # For categorical: show mean target per category
    if df_local[feature].dtype == 'O' or df_local[feature].nunique() < 10:
        grp = df_local.groupby(feature)[target].mean().sort_values(ascending=False)
        fig, ax = plt.subplots()
        grp.plot(kind='bar', ax=ax)
        ax.set_ylabel('Mean ' + target)
        st.pyplot(fig)
    else:
        # numeric: bin and show
        binned = pd.cut(df_local[feature].dropna(), bins=10)
        grp = df_local.groupby(binned)[target].mean()
        fig, ax = plt.subplots()
        grp.plot(kind='bar', ax=ax)
        ax.set_ylabel('Mean ' + target)
        st.pyplot(fig)

# ---------------- Home ----------------
if page == 'Home':
    st.title('Medical Claims AI Dashboard')
    st.write('Use this app to run EDA, RandomForest models, anomaly detection and simple forecasts. Only Diagnosis 1 is used for diagnosis information.')
    if df is None:
        st.info('Upload a dataset (CSV/XLSX) using the sidebar to begin.')
    else:
        st.success('Data loaded and preprocessed')
        st.dataframe(df.head())

# ---------------- EDA ----------------
elif page == 'EDA':
    st.title('Exploratory Data Analysis')
    if df is None:
        st.warning('Upload data first')
    else:
        st.subheader('Basic info')
        st.write(df.describe(include='all'))

        if 'Diagnosis 1' in df.columns:
            st.subheader('Top Diagnosis 1')
            topd = df['Diagnosis 1'].fillna('missing').value_counts().head(20)
            fig, ax = plt.subplots()
            topd.plot(kind='bar', ax=ax)
            st.pyplot(fig)

        if 'Total Bill (RM)' in df.columns:
            st.subheader('Total Bill distribution')
            fig, ax = plt.subplots()
            sns.histplot(df['Total Bill (RM)'].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

        # Correlation heatmap between Age, Diagnosis1 (encoded), Total Bill
        if 'Age' in df.columns and 'Total Bill (RM)' in df.columns and 'Diagnosis 1' in df.columns:
            st.subheader('Correlation between Age, Diagnosis1 (encoded) and Total Bill')
            df_corr = df[['Age','Diagnosis 1','Total Bill (RM)']].dropna()
            if not df_corr.empty:
                df_corr['Diagnosis 1 Encoded'] = LabelEncoder().fit_transform(df_corr['Diagnosis 1'].astype(str))
                corr = df_corr[['Age','Diagnosis 1 Encoded','Total Bill (RM)']].corr()
                st.write(corr)
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

# ---------------- AI Models (classification) ----------------
elif page == 'AI Models':
    st.title('Claim Approval Prediction (RandomForest)')
    if df is None:
        st.warning('Upload data first')
    elif 'Case Status' not in df.columns:
        st.error('Column "Case Status" not found')
    else:
        data = df.dropna(subset=['Diagnosis 1','Case Status']).copy()
        # feature engineering & encoding
        data['Diagnosis 1 Encoded'] = LabelEncoder().fit_transform(data['Diagnosis 1'].astype(str))
        data['Clinic State Encoded'] = LabelEncoder().fit_transform(data['Clinic State'].astype(str)) if 'Clinic State' in data.columns else 0
        data['Age'] = data['Age']
        # optional numeric
        if 'Total Bill (RM)' in data.columns:
            data['Total Bill (RM)'] = pd.to_numeric(data['Total Bill (RM)'], errors='coerce')
        feature_cols = ['Diagnosis 1 Encoded','Age','Clinic State Encoded']
        if 'Total Bill (RM)' in data.columns:
            feature_cols.append('Total Bill (RM)')

        X = data[feature_cols].fillna(0)
        y = LabelEncoder().fit_transform(data['Case Status'].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.metric('Accuracy', f'{acc*100:.2f}%')

        st.subheader('Built-in Feature Importances')
        fi_df = pd.DataFrame({'feature': feature_cols, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x='importance', y='feature', data=fi_df, ax=ax)
        st.pyplot(fig)

        st.subheader('Permutation Importances (SHAP-like)')
        try:
            perm = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
            perm_df = pd.DataFrame({'feature': feature_cols, 'perm_importance': perm.importances_mean}).sort_values('perm_importance', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x='perm_importance', y='feature', data=perm_df, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Permutation importance failed: {e}')

        st.subheader('Feature effects (approximate)')
        # show effects for top 3 features
        top_feats = fi_df['feature'].tolist()[:3]
        for f in top_feats:
            st.write('Feature:', f)
            show_feature_effects(data.assign(**{c: data[c] for c in feature_cols}), f, 'Case Status Encoded' if 'Case Status' in data.columns else y, problem='classification')

        st.subheader('Download predictions')
        out = X_test.copy()
        out['predicted'] = preds
        csv = out.to_csv(index=False).encode('utf-8')
        st.download_button('Download Predictions (classification)', csv, 'predictions_classification.csv', 'text/csv')

# ---------------- Cost Prediction (regression) ----------------
elif page == 'Cost Prediction':
    st.title('Total Bill Prediction (RandomForestRegressor)')
    if df is None:
        st.warning('Upload data first')
    elif 'Total Bill (RM)' not in df.columns:
        st.error('Column "Total Bill (RM)" not found')
    else:
        data = df.dropna(subset=['Diagnosis 1']).copy()
        data['Total Bill (RM)'] = pd.to_numeric(data['Total Bill (RM)'], errors='coerce')
        data = data.dropna(subset=['Total Bill (RM)'])

        data['Diagnosis 1 Encoded'] = LabelEncoder().fit_transform(data['Diagnosis 1'].astype(str))
        data['Clinic State Encoded'] = LabelEncoder().fit_transform(data['Clinic State'].astype(str)) if 'Clinic State' in data.columns else 0
        data['Age'] = data['Age']

        feature_cols = ['Diagnosis 1 Encoded','Age','Clinic State Encoded']
        X = data[feature_cols].fillna(0)
        y = data['Total Bill (RM)']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = RandomForestRegressor(random_state=42)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        st.metric('MAE', f'RM {mae:,.2f}')
        st.metric('RMSE', f'RM {rmse:,.2f}')

        st.subheader('Built-in Feature Importances (regression)')
        fi_df = pd.DataFrame({'feature': feature_cols, 'importance': reg.feature_importances_}).sort_values('importance', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x='importance', y='feature', data=fi_df, ax=ax)
        st.pyplot(fig)

        st.subheader('Permutation Importances (regression)')
        try:
            perm = permutation_importance(reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
            perm_df = pd.DataFrame({'feature': feature_cols, 'perm_importance': perm.importances_mean}).sort_values('perm_importance', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x='perm_importance', y='feature', data=perm_df, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Permutation importance failed: {e}')

        st.subheader('Feature effects (approximate)')
        top_feats = fi_df['feature'].tolist()[:3]
        for f in top_feats:
            st.write('Feature:', f)
            show_feature_effects(data.assign(**{c: data[c] for c in feature_cols}), f, 'Total Bill (RM)', problem='regression')

        st.subheader('Download predictions (regression)')
        out = X_test.copy()
        out['predicted_cost'] = preds
        csv = out.to_csv(index=False).encode('utf-8')
        st.download_button('Download Predictions (regression)', csv, 'predictions_regression.csv', 'text/csv')

# ---------------- Anomaly detection & Forecast ----------------
elif page == 'Anomaly & Forecast':
    st.title('Anomaly Detection and Monthly Forecast')
    if df is None:
        st.warning('Upload data first')
    else:
        adf = df.copy()
        # Prepare numeric features
        if 'Total Bill (RM)' in adf.columns:
            adf['Total Bill (RM)'] = pd.to_numeric(adf['Total Bill (RM)'], errors='coerce')
        if 'No. of MC Days' in adf.columns:
            adf['No. of MC Days'] = pd.to_numeric(adf['No. of MC Days'], errors='coerce')
        adf['Diagnosis 1 Encoded'] = LabelEncoder().fit_transform(adf['Diagnosis 1'].astype(str)) if 'Diagnosis 1' in adf.columns else 0
        adf['Clinic State Encoded'] = LabelEncoder().fit_transform(adf['Clinic State'].astype(str)) if 'Clinic State' in adf.columns else 0
        adf['Age'] = adf['Age']

        features_anom = ['Total Bill (RM)','Diagnosis 1 Encoded','Clinic State Encoded']
        if 'No. of MC Days' in adf.columns:
            features_anom.append('No. of MC Days')
        if 'Age' in adf.columns:
            features_anom.append('Age')

        adf_anom = adf[features_anom].fillna(0)

        contamination = st.slider('Anomaly contamination proportion', 0.001, 0.1, 0.01)
        iso = IsolationForest(contamination=contamination, random_state=42)
        try:
            iso.fit(adf_anom)
            adf['anomaly_score'] = iso.decision_function(adf_anom)
            adf['anomaly'] = iso.predict(adf_anom) == -1
            st.write('Anomalies detected:', int(adf['anomaly'].sum()))
            st.dataframe(adf[adf['anomaly']].sort_values('anomaly_score').head(200))
            st.download_button('Download anomalies CSV', adf[adf['anomaly']].to_csv(index=False).encode('utf-8'), 'anomalies.csv', 'text/csv')
        except Exception as e:
            st.error(f'Anomaly detection failed: {e}')

        st.markdown('---')
        st.subheader('Monthly Total Bill Forecast (simple linear trend)')
        if 'Visit Date' not in adf.columns or 'Total Bill (RM)' not in adf.columns:
            st.info('Visit Date and Total Bill (RM) required for forecasting')
        else:
            ts = adf.dropna(subset=['Visit Date']).copy()
            ts['Visit Date'] = pd.to_datetime(ts['Visit Date'], errors='coerce')
            ts = ts.dropna(subset=['Visit Date'])
            ts['YearMonth'] = ts['Visit Date'].dt.to_period('M').astype(str)
            monthly = ts.groupby('YearMonth')['Total Bill (RM)'].sum().reset_index()
            monthly['Total Bill (RM)'] = pd.to_numeric(monthly['Total Bill (RM)'], errors='coerce').fillna(0)
            monthly = monthly.sort_values('YearMonth').reset_index(drop=True)
            monthly['month_idx'] = np.arange(len(monthly))

            X_time = monthly[['month_idx']]
            y_time = monthly['Total Bill (RM)']
            lr = LinearRegression()
            lr.fit(X_time, y_time)

            n_forecast = st.slider('Forecast months', 1, 24, 6)
            future_idx = np.arange(len(monthly), len(monthly) + n_forecast)
            future_pred = lr.predict(future_idx.reshape(-1,1))

            last_period = pd.Period(monthly['YearMonth'].iloc[-1])
            future_months = [(last_period + i).strftime('%Y-%m') for i in range(1, n_forecast+1)]
            forecast_df = pd.DataFrame({'YearMonth': future_months, 'Forecasted Total Bill (RM)': future_pred})

            # Plot
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(monthly['YearMonth'], monthly['Total Bill (RM)'], label='Historical')
            ax.plot(forecast_df['YearMonth'], forecast_df['Forecasted Total Bill (RM)'], label='Forecast')
            ax.set_xticks(list(range(0, len(monthly)+len(forecast_df), max(1, (len(monthly)+len(forecast_df))//10))))
            ax.set_xticklabels(list(monthly['YearMonth']) + list(forecast_df['YearMonth']), rotation=45)
            ax.set_ylabel('Total Bill (RM)')
            ax.legend()
            st.pyplot(fig)
            st.download_button('Download forecast CSV', forecast_df.to_csv(index=False).encode('utf-8'), 'forecast.csv', 'text/csv')

# ---------------- About ----------------
elif page == 'About':
    st.title('About')
    st.write('This app runs RandomForest models to explore relationships between Age, Diagnosis 1, Clinic State, and Total Bill.')
    st.write('It provides feature importance, permutation importance (SHAP-like), anomaly detection (IsolationForest), and a simple linear forecast for monthly totals.')

# End of app
