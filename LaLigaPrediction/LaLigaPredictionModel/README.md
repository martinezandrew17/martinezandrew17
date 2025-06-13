# ⚽ La Liga Predictor

This project is a fully automated machine learning dashboard that predicts the **La Liga champion** and **top scorers** for the next 5 seasons (2025–2029), powered by XGBoost and Streamlit.

![Streamlit App Screenshot](https://placehold.co/800x400?text=La+Liga+Predictor+Dashboard)

---

## 🚀 Live Demo

👉 [Launch the App on Streamlit Cloud](https://your-username-your-repo-name.streamlit.app)

---

## 💡 Features

- 🏆 Predicts the **La Liga champion** each season
- 🎯 Displays **top 10 goal scorers** per season
- 📊 Interactive **bar chart visualization** of predictions
- 📁 Synthetic data generation for teams and players
- 🧠 Machine Learning with **XGBoost Classifier & Regressor**

---

## 🛠️ How It Works

### 1. Simulate Synthetic Data

```bash
python src/simulation/simulate_future_teams.py
python src/simulation/simulate_future_players.py
```
