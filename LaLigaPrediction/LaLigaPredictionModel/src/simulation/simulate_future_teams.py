# src/simulation/simulate_future_teams.py

import pandas as pd
import numpy as np
import os

def generate_synthetic_team_data(start_year=2025, seasons=5):
    np.random.seed(42)
    teams = [
        "Real Madrid", "Barcelona", "Atlético Madrid", "Girona", "Athletic Club",
        "Real Sociedad", "Real Betis", "Villarreal", "Valencia", "Sevilla",
        "Getafe", "Osasuna", "Las Palmas", "Almería", "Granada",
        "Mallorca", "Rayo Vallecano", "Cádiz", "Alavés", "Celta Vigo"
    ]

    data = []
    for season in range(start_year, start_year + seasons):
        for team in teams:
            wins = np.random.randint(12, 27)
            draws = np.random.randint(5, 15)
            losses = 38 - wins - draws
            goals_scored = np.random.randint(40, 90)
            goals_against = np.random.randint(30, 80)
            goal_diff = goals_scored - goals_against
            points = wins * 3 + draws
            data.append({
                "season": season,
                "team": team,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "goals_scored": goals_scored,
                "goals_against": goals_against,
                "goal_diff": goal_diff,
                "points": points
            })

    os.makedirs("data/synthetic", exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv("data/synthetic/future_teams.csv", index=False)
    print("✅ Synthetic future team data saved to data/synthetic/future_teams.csv")
    return df

if __name__ == "__main__":
    generate_synthetic_team_data()