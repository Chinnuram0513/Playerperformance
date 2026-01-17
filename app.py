import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="IPL Player Performance Analysis", layout="wide")

# ----------------------------
# Load data & model
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("player_match_batting_stats.csv", parse_dates=["date"])

bat_df = load_data()
model = joblib.load("final_runs_prediction_model.pkl")

# ----------------------------
# App Title
# ----------------------------
st.title("IPL Player Performance Prediction & Analysis")
st.write("Analyze player performance and predict expected runs based on recent form")

# ----------------------------
# Player Selection
# ----------------------------
player = st.selectbox(
    "Select Player",
    sorted(bat_df["batsman"].unique())
)

player_df = bat_df[bat_df["batsman"] == player].sort_values("date")

# ----------------------------
# Career Summary
# ----------------------------
st.subheader("Career Summary")

col1, col2, col3, col4 = st.columns(4)

matches = player_df.shape[0]
total_runs = player_df["runs_scored"].sum()
avg_runs = round(total_runs / matches, 2)
avg_sr = round(player_df["strike_rate"].mean(), 2)

col1.metric("Matches", matches)
col2.metric("Total Runs", total_runs)
col3.metric("Average Runs", avg_runs)
col4.metric("Strike Rate", avg_sr)

# ----------------------------
# Recent Performance
# ----------------------------
st.subheader("Recent Performance")

last_5 = player_df.tail(5)
last_10 = player_df.tail(10)

runs_last_5 = round(last_5["runs_scored"].mean(), 2)
runs_last_10 = round(last_10["runs_scored"].mean(), 2)
sr_last_5 = round(last_5["strike_rate"].mean(), 2)
sr_last_10 = round(last_10["strike_rate"].mean(), 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Runs (Last 5)", runs_last_5)
c2.metric("Avg Runs (Last 10)", runs_last_10)
c3.metric("SR (Last 5)", sr_last_5)
c4.metric("SR (Last 10)", sr_last_10)

# ----------------------------
# Performance Trend
# ----------------------------
st.subheader("Runs Trend Over Matches")
st.line_chart(
    player_df.set_index("date")["runs_scored"]
)

# ----------------------------
# Prediction Section
# ----------------------------
st.subheader("Predicted Next Match Runs")

features = np.array([[runs_last_5, runs_last_10, sr_last_5, sr_last_10]])
prediction = model.predict(features)[0]

st.success(f"Expected Runs: {prediction:.2f}")
