import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Churn Prediction App", layout="wide")

# -----------------------------
# SAFE MODEL LOADER (IMPORTANT)
# -----------------------------
def load_model():
    try:
        with open("churn_pipeline.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error("❌ Failed to load model")
        st.error(str(e))
        return None

model = load_model()

st.title("📊 Customer Churn Prediction App")

if model is None:
    st.stop()

# -----------------------------
# INPUT FIELDS
# -----------------------------
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])

with col2:
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", 0, 72)
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])

with col3:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    charges = st.number_input("Monthly Charges", 0.0, 200.0)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phoneservice,
        "InternetService": internet,
        "Contract": contract,
        "MonthlyCharges": charges
    }])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.success(f"🔍 **Prediction:** {'Churn' if prediction == 1 else 'No Churn'}")
    st.info(f"📈 **Churn Probability:** {prob:.2f}")
