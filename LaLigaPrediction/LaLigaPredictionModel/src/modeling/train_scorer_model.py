# src/modeling/train_scorer_model.py

import pandas as pd
import xgboost as xgb
import os
import pickle

def load_historical_player_data():
    path = "data/raw/top_scorers_historical.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return df

def train():
    df = pd.read_csv("data/raw/top_scorers_historical.csv")
    df.columns = df.columns.str.lower()

    X = df[["matches", "minutes", "goals_per_match", "minutes_per_goal"]]
    y = df["goals"]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    with open("models/top_scorer_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Top scorer model trained and saved.")

if __name__ == "__main__":
    train()