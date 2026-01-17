import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="IPL Player Performance Analysis",
    page_icon="🏏",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🏏 IPL Player Performance Prediction & Analysis")
st.markdown(
    "Analyze **player-wise IPL performance** and predict **expected runs** based on recent form."
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("player_match_batting_stats.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# ---------------- LOAD MODEL ----------------
model = joblib.load("final_runs_prediction_model.pkl")

# ---------------- PLAYER SELECTION ----------------
player = st.selectbox(
    "Select Player",
    sorted(df["batsman"].unique())
)

player_df = df[df["batsman"] == player].sort_values("date")

# ---------------- CAREER SUMMARY ----------------
st.subheader("📊 Career Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Matches", len(player_df))
c2.metric("Total Runs", int(player_df["runs_scored"].sum()))
c3.metric("Avg Runs", round(player_df["runs_scored"].mean(), 2))
c4.metric("Avg Strike Rate", round(player_df["strike_rate"].mean(), 2))

# ---------------- RECENT FORM ----------------
st.subheader("🔥 Recent Form")

last_5 = player_df.tail(5)
last_10 = player_df.tail(10)

r1, r2, r3, r4 = st.columns(4)

r1.metric("Runs (Last 5)", round(last_5["runs_scored"].mean(), 2))
r2.metric("Runs (Last 10)", round(last_10["runs_scored"].mean(), 2))
r3.metric("SR (Last 5)", round(last_5["strike_rate"].mean(), 2))
r4.metric("SR (Last 10)", round(last_10["strike_rate"].mean(), 2))

# ---------------- RUNS TREND ----------------
st.subheader("📈 Runs Trend Over Matches")

fig_runs = px.line(
    player_df,
    x="date",
    y="runs_scored",
    markers=True,
    title="Runs Scored per Match",
    color_discrete_sequence=["#00E5FF"]
)

st.plotly_chart(fig_runs, use_container_width=True)

# ---------------- LAST 10 MATCH BAR ----------------
st.subheader("🏏 Last 10 Match Performance")

fig_bar = px.bar(
    last_10,
    x="date",
    y="runs_scored",
    text="runs_scored",
    color="runs_scored",
    color_continuous_scale="Viridis",
    title="Runs in Last 10 Matches"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- STRIKE RATE TREND ----------------
st.subheader("⚡ Strike Rate Trend")

fig_sr = px.line(
    player_df,
    x="date",
    y="strike_rate",
    markers=True,
    color_discrete_sequence=["#FFD700"],
    title="Strike Rate Over Time"
)

st.plotly_chart(fig_sr, use_container_width=True)

# ---------------- PREDICTION ----------------
st.subheader("🔮 Run Prediction for Next Match")

features = np.array([[
    last_5["runs_scored"].mean(),
    last_10["runs_scored"].mean(),
    last_5["strike_rate"].mean(),
    last_10["strike_rate"].mean()
]])

predicted_runs = model.predict(features)[0]

st.success(f"🏆 **Expected Runs: {round(predicted_runs, 1)}**")

# ---------------- DATA PREVIEW ----------------
with st.expander("📁 View Player Match Data"):
    st.dataframe(player_df.tail(10))
