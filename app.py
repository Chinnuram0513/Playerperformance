import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="IPL Run Predictor", layout="centered")

st.title("🏏 IPL Player Run Prediction")
st.write("Predict expected runs based on recent performance")

model = joblib.load("final_runs_prediction_model.pkl")

runs_last_5 = st.number_input("Average Runs (Last 5 Matches)", 0.0, 100.0, 30.0)
runs_last_10 = st.number_input("Average Runs (Last 10 Matches)", 0.0, 100.0, 35.0)
sr_last_5 = st.number_input("Strike Rate (Last 5 Matches)", 0.0, 300.0, 130.0)
sr_last_10 = st.number_input("Strike Rate (Last 10 Matches)", 0.0, 300.0, 125.0)

if st.button("Predict Runs"):
    features = np.array([[runs_last_5, runs_last_10, sr_last_5, sr_last_10]])
    prediction = model.predict(features)[0]
    st.success(f"🏆 Predicted Runs: {prediction:.2f}")
