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

# ---------------- TITLE & IMAGE ----------------
st.title("🏏 IPL Player Performance Prediction & Analysis")
st.write("Analyze player performance and predict expected runs based on recent form")

st.image(
    "https://images.unsplash.com/photo-1593766827228-8737b4534aa6",
    use_container_width=True
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("player_match_batting_stats.csv")

df = load_data()

# ---------------- LOAD MODEL ----------------
model = joblib.load("final_runs_prediction_model.pkl")

# ---------------- PLAYER SELECTION ----------------
player = st.selectbox("Select Player", sorted(df["player_name"].unique()))
player_df = df[df["player_name"] == player].sort_values("match_date")

# ---------------- CAREER SUMMARY ----------------
st.subheader("📊 Career Summary")

total_matches = len(player_df)
total_runs = player_df["runs_scored"].sum()
avg_runs = player_df["runs_scored"].mean()
avg_sr = player_df["strike_rate"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches", total_matches)
c2.metric("Total Runs", total_runs)
c3.metric("Average Runs", round(avg_runs, 2))
c4.metric("Strike Rate", round(avg_sr, 2))

# ---------------- RECENT PERFORMANCE ----------------
st.subheader("🔥 Recent Performance")

last_5 = player_df.tail(5)
last_10 = player_df.tail(10)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Avg Runs (Last 5)", round(last_5["runs_scored"].mean(), 2))
r2.metric("Avg Runs (Last 10)", round(last_10["runs_scored"].mean(), 2))
r3.metric("SR (Last 5)", round(last_5["strike_rate"].mean(), 2))
r4.metric("SR (Last 10)", round(last_10["strike_rate"].mean(), 2))

# ---------------- RUNS TREND ----------------
st.subheader("📈 Runs Trend Over Matches")

fig_runs = px.line(
    player_df,
    x="match_date",
    y="runs_scored",
    markers=True,
    title="Runs Scored Per Match",
    color_discrete_sequence=["#00FFAB"]
)

fig_runs.update_layout(
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font_color="white"
)

st.plotly_chart(fig_runs, use_container_width=True)

# ---------------- LAST 10 MATCHES BAR ----------------
st.subheader("🏏 Last 10 Matches Performance")

fig_bar = px.bar(
    last_10,
    x="match_date",
    y="runs_scored",
    text="runs_scored",
    color="runs_scored",
    color_continuous_scale="Turbo"
)

fig_bar.update_layout(
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font_color="white"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- STRIKE RATE TREND ----------------
st.subheader("⚡ Strike Rate Trend")

fig_sr = px.line(
    player_df,
    x="match_date",
    y="strike_rate",
    markers=True,
    color_discrete_sequence=["#FFD700"]
)

fig_sr.update_layout(
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font_color="white"
)

st.plotly_chart(fig_sr, use_container_width=True)

# ---------------- PREDICTION ----------------
st.subheader("🔮 Predicted Performance")

features = np.array([[
    last_5["runs_scored"].mean(),
    last_10["runs_scored"].mean(),
    last_5["strike_rate"].mean(),
    last_10["strike_rate"].mean()
]])

predicted_runs = model.predict(features)[0]

st.success(f"🏆 Expected Runs in Next Match: **{round(predicted_runs, 1)}**")

# ---------------- FOOTER IMAGE ----------------
st.image(
    "https://images.unsplash.com/photo-1623944891021-4a6b7c1cfc6f",
    caption="IPL Match Action",
    use_container_width=True
)
