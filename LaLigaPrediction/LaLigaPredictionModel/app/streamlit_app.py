import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from src.utils.helpers import load_model, predict_champion, predict_scorers
from src.feature_engineering.player_features import load_top_scorer_data

st.set_page_config(page_title="La Liga Predictor", layout="centered")
st.title("⚽ La Liga Predictor")

st.header("🏆 Champion Prediction")
champ_model = load_model("models/champion_model.pkl")

data = {
    "wins": st.number_input("Wins", 0, 40),
    "draws": st.number_input("Draws", 0, 40),
    "losses": st.number_input("Losses", 0, 40),
    "goals_scored": st.number_input("Goals Scored", 0, 200),
    "goals_against": st.number_input("Goals Conceded", 0, 200),
    "goal_diff": st.number_input("Goal Difference", -100, 100),
    "points": st.number_input("Points", 0, 120)
}

if st.button("Predict Champion Status"):
    input_df = pd.DataFrame([data])
    result = predict_champion(champ_model, input_df)
    if result[0] == 1:
        st.success("This team is predicted to be the champion! 🏆")
    else:
        st.info("This team is not predicted to be the champion.")

st.divider()
st.header("🎯 Top Scorer Prediction")
scorer_model = load_model("models/top_scorer_model.pkl")

if st.button("Load and Predict Top Scorers"):
    df = load_top_scorer_data()
    X = df[['matches', 'minutes', 'goals_per_match', 'minutes_per_goal']]
    df["predicted_goals"] = predict_scorers(scorer_model, X)["predicted_goals"]
    st.dataframe(df[['player', 'team', 'predicted_goals']].head(10))
