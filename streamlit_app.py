import streamlit as st
import pandas as pd
import joblib
import os
import traceback

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction")

MODEL_PATH = "churn_pipeline.pkl"

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model file not found at path: {path}")
        st.stop()
    try:
        # joblib used since model was saved with joblib/pickle-compatible pipeline
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error("❌ Failed to load model. See error below:")
        st.error(str(e))
        # print detailed traceback in logs for debugging
        st.write("**Debug (traceback):**")
        st.text(traceback.format_exc())
        st.stop()

model = load_model()

st.markdown("### Enter customer details (use realistic values)")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen (0 = No, 1 = Yes)", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=240, value=12, step=1)

with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.5)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=500.0, step=1.0)

if st.button("Predict"):
    # build dataframe matching the model's expected columns
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "InternetService": InternetService,
        "Contract": Contract,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("Result")
        if pred == 1:
            st.error(f"❌ Customer is likely to CHURN. Probability: {prob:.3f}" if prob is not None else "❌ Customer likely to CHURN.")
        else:
            st.success(f"✅ Customer is NOT likely to churn. Probability: {prob:.3f}" if prob is not None else "✅ Customer NOT likely to churn.")
    except Exception as e:
        st.error("Prediction failed. Check the model and input columns.")
        st.write(str(e))
        st.text(traceback.format_exc())
