import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Convert inputs to dataframe
data = {
    "SeniorCitizen": senior,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "gender_Male": 1 if gender == "Male" else 0
}

input_df = pd.DataFrame([data])

# Match training columns if needed
model_features = getattr(model, "feature_names_in_", [])
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

if st.button("Predict"):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Customer is likely to leave the service.")
    else:
        st.success("Customer is likely to stay with the service.")