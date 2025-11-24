import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------
# Load trained pipeline
# ------------------------------
@st.cache_resource
def load_model():
    with open("churn_pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict whether they will churn.")

# ------------------------------
# Input Layout
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

# ------------------------------
# Submit Button
# ------------------------------
if st.button("Predict Churn"):
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "InternetService": internet_service,
        "Contract": contract,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([input_dict])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")
    if pred == 1:
        st.error(f"❌ Customer is likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn (Probability: {prob:.2f})")
