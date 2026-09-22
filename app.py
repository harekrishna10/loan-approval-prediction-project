# app.py — Loan Approval Prediction App
import streamlit as st
import joblib
import pandas as pd

# Load the trained model pipeline (scaler + SVM)
model = joblib.load("loan_model.pkl")

st.title("Loan Approval Prediction")
st.write("Enter the applicant details to predict whether the loan will be approved.")

# --- User inputs ---
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)
loan_term = st.number_input("Loan Amount Term (in months)", min_value=0, value=360)
credit_history = st.selectbox("Credit History meets guidelines", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --- Prediction ---
if st.button("Predict"):
    # Convert inputs exactly the same way the training data was encoded
    input_df = pd.DataFrame([{
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": 3 if dependents == "3+" else int(dependents),
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1 if credit_history == "Yes" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
        "Total_Income": applicant_income + coapplicant_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Loan Approved ✅ (confidence: {probability:.1%})")
    else:
        st.error(f"Loan Rejected ❌ (approval probability: {probability:.1%})")
