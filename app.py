import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="IPL Player Performance Analytics",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL Player Performance Analytics & Prediction")
st.caption("Batting • Bowling • Player Comparison • ML Insights")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    bat = pd.read_csv("player_match_batting_stats.csv")
    bowl = pd.read_csv("player_match_bowling_stats.csv")

    if "date" in bat.columns:
        bat["date"] = pd.to_datetime(bat["date"], errors="coerce")

    return bat, bowl

bat_df, bowl_df = load_data()

# ================= LOAD MODEL =================
model = joblib.load("final_runs_prediction_model.pkl")

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏏 Batting Analysis",
    "🎯 Bowling Analysis",
    "⚔ Player Comparison",
    "📊 Model Insights"
])

# =====================================================
# 🏏 BATSMAN ANALYSIS
# =====================================================
with tab1:
    st.subheader("🏏 Batting Performance")

    batsman = st.selectbox(
        "Select Batsman",
        sorted(bat_df["batsman"].dropna().unique())
    )

    df = bat_df[bat_df["batsman"] == batsman].sort_values("date")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", len(df))
    c2.metric("Total Runs", int(df["runs_scored"].sum()))
    c3.metric("Avg Runs", round(df["runs_scored"].mean(), 2))
    c4.metric("Avg Strike Rate", round(df["strike_rate"].mean(), 2))

    fig_runs = px.line(
        df,
        x="date",
        y="runs_scored",
        markers=True,
        title="Runs Scored Per Match"
    )
    st.plotly_chart(fig_runs, use_container_width=True)

    # Prediction
    last_5 = df.tail(5)
    last_10 = df.tail(10)

    if len(last_10) >= 5:
        features = np.array([[
            last_5["runs_scored"].mean(),
            last_10["runs_scored"].mean(),
            last_5["strike_rate"].mean(),
            last_10["strike_rate"].mean()
        ]])

        pred = model.predict(features)[0]
        st.success(f"🔮 Expected Runs Next Match: **{round(pred, 1)}**")
    else:
        st.warning("Not enough matches for prediction")

# =====================================================
# 🎯 BOWLING ANALYSIS (FIXED PROPERLY)
# =====================================================
with tab2:
    st.subheader("🎯 Bowling Performance")

    bowler = st.selectbox(
        "Select Bowler",
        sorted(bowl_df["bowler"].dropna().unique())
    )

    df = bowl_df[bowl_df["bowler"] == bowler].sort_values("matchId")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", len(df))
    c2.metric("Total Wickets", int(df["wickets"].sum()))
    c3.metric("Avg Economy", round(df["economy"].mean(), 2))

    # ✅ SAFE bowling strike rate
    if "balls" in df.columns:
        strike_rate = df["balls"].sum() / max(df["wickets"].sum(), 1)
        c4.metric("Strike Rate", round(strike_rate, 2))
    else:
        c4.metric("Strike Rate", "N/A")

    fig_wickets = px.bar(
        df.tail(10),
        x="matchId",
        y="wickets",
        color="wickets",
        title="Wickets in Last 10 Matches"
    )
    st.plotly_chart(fig_wickets, use_container_width=True)

# =====================================================
# ⚔ PLAYER vs PLAYER
# =====================================================
with tab3:
    st.subheader("⚔ Player vs Player Comparison")

    col1, col2 = st.columns(2)

    p1 = col1.selectbox(
        "Player 1",
        sorted(bat_df["batsman"].dropna().unique())
    )
    p2 = col2.selectbox(
        "Player 2",
        sorted(bat_df["batsman"].dropna().unique()),
        index=1
    )

    df1 = bat_df[bat_df["batsman"] == p1]
    df2 = bat_df[bat_df["batsman"] == p2]

    comp_df = pd.DataFrame({
        "Player": [p1, p2],
        "Avg Runs": [df1["runs_scored"].mean(), df2["runs_scored"].mean()],
        "Strike Rate": [df1["strike_rate"].mean(), df2["strike_rate"].mean()]
    })

    fig_comp = px.bar(
        comp_df,
        x="Player",
        y=["Avg Runs", "Strike Rate"],
        barmode="group",
        title="Batting Comparison"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# =====================================================
# 📊 MODEL INSIGHTS
# =====================================================
with tab4:
    st.subheader("📊 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": [
            "Runs (Last 5)",
            "Runs (Last 10)",
            "Strike Rate (Last 5)",
            "Strike Rate (Last 10)"
        ],
        "Importance": model.feature_importances_
    })

    fig_imp = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        color="Importance",
        title="Feature Importance"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "Shows which features most influence the run prediction model."
    )
