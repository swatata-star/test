import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Prediction App", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Enter the customer details below to predict whether the customer will churn.")

# ----------------------
# Load model
# ----------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")

model = load_model()

# ----------------------
# User Input Form
# ----------------------
st.header("📝 Customer Information")

tenure = st.number_input("Tenure (Months):", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges:", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges:", 0.0, 10000.0, 500.0)

gender = st.selectbox("Gender:", ["Male", "Female"])
partner = st.selectbox("Partner:", ["Yes", "No"])
dependents = st.selectbox("Dependents:", ["Yes", "No"])
phone_service = st.selectbox("Phone Service:", ["Yes", "No"])
internet_service = st.selectbox("Internet Service:", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type:", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method:", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

# ----------------------
# Dataframe for Model
# ----------------------
input_data = pd.DataFrame({
    "gender": [gender],
    "Partner": [partner],
    "Dependents": [dependents],
    "PhoneService": [phone_service],
    "InternetService": [internet_service],
    "Contract": [contract],
    "PaymentMethod": [payment_method],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
})

# ----------------------
# Prediction
# ----------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"❌ The customer is **likely to churn**.\n\n**Probability: {probability:.2f}**")
    else:
        st.success(f"✅ The customer is **not likely to churn**.\n\n**Probability of churn: {probability:.2f}**")

    st.markdown("---")
    st.caption("This prediction is based on the trained machine learning model using customer behavior patterns.")

# Streamlit app will be inserted here
