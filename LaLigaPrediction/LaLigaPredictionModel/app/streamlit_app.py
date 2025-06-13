import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os
import altair as alt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.helpers import load_model

st.set_page_config(page_title="La Liga Predictor", layout="wide")
st.title("⚽ La Liga Predictor")

champ_model = load_model("models/champion_model.pkl")
scorer_model = load_model("models/top_scorer_model.pkl")

tdf = pd.read_csv("data/synthetic/future_teams.csv")
pdf = pd.read_csv("data/synthetic/future_players.csv")

seasons = sorted(tdf["season"].unique())
selected_season = st.selectbox("Select Season", seasons, index=0)

left_col, right_col = st.columns(2)

with left_col:
    st.subheader(f"🏆 Predicted Champion - {selected_season}")
    group = tdf[tdf["season"] == selected_season].copy()
    X = group[["wins", "draws", "losses", "goals_scored", "goals_against", "goal_diff", "points"]]
    try:
        group["champion_pred"] = champ_model.predict(X)
        winner = group[group["champion_pred"] == 1]
        if not winner.empty:
            st.markdown(f"### 🥇 {winner.iloc[0]['team']}")
        else:
            top_team = group.sort_values("points", ascending=False).iloc[0]["team"]
            st.markdown(f"### 🥇 {top_team}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
    st.dataframe(group[["team", "wins", "draws", "losses", "points"]].sort_values("points", ascending=False), use_container_width=True)

with right_col:
    st.subheader(f"🎯 Predicted Top Scorers - {selected_season}")
    scorer_group = pdf[pdf["season"] == selected_season].copy()
    Xp = scorer_group[["matches", "minutes", "goals_per_match", "minutes_per_goal"]]
    scorer_group["predicted_goals"] = scorer_model.predict(Xp)
    top_scorers = scorer_group.sort_values("predicted_goals", ascending=False).head(10)
    top_scorers["player"] = top_scorers["player"].str.replace("_", " ").str.replace("Player", "Player ")
    st.dataframe(top_scorers[["player", "team", "predicted_goals"]], use_container_width=True)

    chart = alt.Chart(top_scorers).mark_bar().encode(
        x=alt.X("predicted_goals:Q", title="Goals"),
        y=alt.Y("player:N", sort="-x", title="Player"),
        color=alt.Color("team:N", title="Team")
    ).properties(width=500, height=350)

    st.altair_chart(chart, use_container_width=True)

st.divider()
st.caption("La Liga predictions powered by machine learning (XGBoost) -- Visca El Barça!")