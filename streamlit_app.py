import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Churn Prediction")

# -------------------
# Load Model Safely
# -------------------
def load_model():
    file_path = "churn_pipeline.pkl"
    if not os.path.exists(file_path):
        st.error(f"Model file not found: {file_path}")
        st.stop()

    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error("❌ Failed to load model")
        st.error(str(e))
        st.stop()

model = load_model()

st.title("📊 Customer Churn Prediction")

st.write("Fill the details below:")

# Input layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", 0, 72, 12)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# Prepare data for prediction
data = pd.DataFrame([{
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
}])

if st.button("Predict"):
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        st.error(f"❌ Customer likely to CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer NOT likely to churn (Probability: {prob:.2f})")
