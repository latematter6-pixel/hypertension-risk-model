import streamlit as st
import pickle
import numpy as np

# Загрузка модели и scaler
model = pickle.load(open("hypertension_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Hypertension Risk Prediction")

st.write("Enter patient data:")

age = st.number_input("Age", min_value=1, max_value=120)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
glucose = st.number_input("Glucose Level", min_value=50, max_value=300)
smoking = st.selectbox("Smoking (0=No, 1=Yes)", [0, 1])
alcohol = st.selectbox("Alcohol (0=No, 1=Yes)", [0, 1])
physical_activity = st.selectbox("Physical Activity (0=No, 1=Yes)", [0, 1])

if st.button("Predict"):
    data = np.array([[age, bmi, glucose, smoking, alcohol, physical_activity]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.error("High risk of hypertension")
    else:
        st.success("Low risk of hypertension")
