import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("🐎 Horse Race Outcome Predictor")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload horse race data (CSV format)", type=["csv"])

# Load models and tools
try:
    log_model = joblib.load("logistic_regression_model.pkl")
    xgb_model = joblib.load("xgboost_model.pkl")
    tuned_xgb = joblib.load("tuned_xgboost_model.pkl")
    stacked_model = joblib.load("stacked_meta_model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    final_columns = joblib.load("final_columns.pkl")

except Exception as e:
    st.error(f"❌ Error loading models or preprocessing tools:\n\n{e}")
    st.stop()

# Clean column names for XGBoost compatibility
def standardize_column_names(df):
    replacements = {
        '<=': 'LE=',
        '>=': 'GE=',
        '<': 'LT=',
        '>': 'GT=',
        '=': '',
        '=>': 'GE=',
        '=<': 'LE=',
    }
    new_cols = []
    for col in df.columns:
        new_col = col
        for old, new in replacements.items():
            new_col = new_col.replace(old, new)
        new_cols.append(new_col)
    df.columns = new_cols
    return df

# Process the uploaded file
if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)

        # Ensure required columns are present
        missing_cols = set(final_columns) - set(user_df.columns)
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        # Keep only required columns and order correctly
        user_input = user_df[final_columns]

        # Handle missing values
        user_input_imputed = pd.DataFrame(imputer.transform(user_input), columns=final_columns)

        # Scale for logistic regression
        user_input_scaled = scaler.transform(user_input_imputed)

        # Standardize names for XGBoost
        xgb_input = standardize_column_names(user_input_imputed.copy())

        # Predict probabilities (Win Today) for all models
        log_pred = log_model.predict_proba(user_input_scaled)[:, 1]
        xgb_pred = xgb_model.predict_proba(xgb_input)[:, 1]
        tuned_xgb_pred = tuned_xgb.predict_proba(xgb_input)[:, 1]

        # Stacked model uses outputs of LR and XGB as input
        stacked_input = np.column_stack((log_pred, xgb_pred))
        stacked_pred = stacked_model.predict_proba(stacked_input)[:, 1]

        # Build a result DataFrame with % formatting
        result_df = pd.DataFrame({
            "Logistic Regression": (log_pred * 100).round(2),
            "XGBoost": (xgb_pred * 100).round(2),
            "Tuned XGBoost": (tuned_xgb_pred * 100).round(2),
            "Stacked Model": (stacked_pred * 100).round(2)
        })

        # Display results
        st.success("✅ Prediction Results (Win Probability %)")
        st.dataframe(result_df.style.format("{:.2f}%"))

        st.markdown("📌 Each row represents a horse's predicted chance of winning today.")

    except Exception as e:
        st.error(f"❌ Error processing the file:\n\n{e}")
