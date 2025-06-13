# src/simulation/simulate_future_players.py

import os
import pandas as pd
import numpy as np

def generate_synthetic_player_data(start_year=2025, seasons=5):
    np.random.seed(42)
    teams = [
        "Real Madrid", "Barcelona", "Atlético Madrid", "Girona", "Athletic Club",
        "Real Sociedad", "Real Betis", "Villarreal", "Valencia", "Sevilla",
        "Getafe", "Osasuna", "Las Palmas", "Almería", "Granada",
        "Mallorca", "Rayo Vallecano", "Cádiz", "Alavés", "Celta Vigo"
    ]

    data = []
    for year in range(start_year, start_year + seasons):
        for team in teams:
            for i in range(1, 26):  # simulate 25 players per team
                player = f"{team.split()[0]}_Player{i}"
                matches = np.random.randint(0, 38)
                minutes = matches * np.random.randint(10, 95)
                goals = np.random.poisson(lam=matches * 0.3)
                gpm = goals / matches if matches else 0
                mpg = minutes / goals if goals else minutes
                data.append({
                    "season": year,
                    "player": player,
                    "team": team,
                    "matches": matches,
                    "minutes": minutes,
                    "goals": goals,
                    "goals_per_match": gpm,
                    "minutes_per_goal": mpg
                })

    os.makedirs("data/synthetic", exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv("data/synthetic/future_players.csv", index=False)
    print("✅ Synthetic static player data saved to data/synthetic/future_players.csv")
    return df

if __name__ == "__main__":
    generate_synthetic_player_data()