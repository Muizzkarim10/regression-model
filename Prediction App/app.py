import streamlit as st
import pandas as pd
import joblib

# Load models and selected features
lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
selected_columns = joblib.load("selected_columns.pkl")

st.set_page_config(page_title="Horse Win Predictor", page_icon="🐎")
st.title("🏇 Horse Win Probability Predictor")

st.markdown("Upload a CSV **or** enter horse data manually below.")

# --- File Upload Section ---
st.header("📁 Upload CSV or Excel File")
uploaded_file = st.file_uploader("Upload a CSV or XLSX file with horse data", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read based on file extension
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file type.")
            df = None

        if df is not None:
            missing = [col for col in selected_columns if col not in df.columns]
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
            else:
                X_input = df[selected_columns]

                # Predict with all models
                lr_probs = lr_model.predict_proba(X_input)[:, 1] * 100
                rf_probs = rf_model.predict_proba(X_input)[:, 1] * 100
                xgb_probs = xgb_model.predict_proba(X_input)[:, 1] * 100

                df["LR Win Probability (%)"] = lr_probs.round(2)
                df["RF Win Probability (%)"] = rf_probs.round(2)
                df["XGB Win Probability (%)"] = xgb_probs.round(2)

                st.success("✅ Predictions generated!")
                st.subheader("🔮 Win Probabilities")
                st.dataframe(df[["LR Win Probability (%)", "RF Win Probability (%)", "XGB Win Probability (%)"]])

                # Export as CSV
                output_csv = df.to_csv(index=False)
                st.download_button("📥 Download Predictions", output_csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

# --- Manual Input Section ---
st.header("✍️ Manual Entry")

with st.form("manual_input_form"):
    st.subheader("Enter values for one horse")
    manual_data = {}
    for col in selected_columns:
        manual_data[col] = st.number_input(f"{col}", value=0.0)
    
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([manual_data])
        lr_prob = lr_model.predict_proba(input_df)[0][1] * 100
        rf_prob = rf_model.predict_proba(input_df)[0][1] * 100
        xgb_prob = xgb_model.predict_proba(input_df)[0][1] * 100

        st.success("🔮 Win Probabilities")
        st.write(f"📘 Logistic Regression: **{lr_prob:.2f}%**")
        st.write(f"🌲 Random Forest: **{rf_prob:.2f}%**")
        st.write(f"⚡ XGBoost: **{xgb_prob:.2f}%**")
