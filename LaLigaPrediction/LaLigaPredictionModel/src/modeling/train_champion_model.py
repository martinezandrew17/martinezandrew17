# src/modeling/train_champion_model.py

import pandas as pd
import xgboost as xgb
import os
import pickle

def load_historical_team_data():
    path = "data/raw/partidos_LaLiga.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"temporada": "season", "equipo local": "home_team", "equipo visitante": "away_team"})
    return df

def process_team_features():
    future_df = pd.read_csv("data/synthetic/future_teams.csv")
    return future_df.drop(columns=["season", "team"], errors="ignore")

def label_champions(team_df):
    team_df = team_df.copy()
    labeled = []
    for season, group in team_df.groupby("season"):
        top = group.sort_values("points", ascending=False).head(1)
        group["champion"] = group["team"].isin(top["team"]).astype(int)
        labeled.append(group)
    return pd.concat(labeled)

def train():
    df = pd.read_csv("data/raw/historical_team_stats.csv")
    df = label_champions(df)

    X = df[["wins", "draws", "losses", "goals_scored", "goals_against", "goal_diff", "points"]]
    y = df["champion"]

    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    with open("models/champion_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Champion model trained and saved.")

if __name__ == "__main__":
    train()