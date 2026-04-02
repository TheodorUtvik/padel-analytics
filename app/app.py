"""
Padel Analytics — Streamlit Dashboard
Run with: streamlit run app/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import streamlit as st

from components.sidebar import render_sidebar

PROCESSED = Path(__file__).parents[1] / "data" / "processed"

st.set_page_config(
    page_title="Padel Analytics",
    page_icon="🎾",
    layout="wide",
)

render_sidebar()

# --- Header ---
st.title("🎾 Padel Analytics")
st.markdown(
    "An end-to-end data science project — collecting professional padel match data, "
    "engineering predictive features, and serving ML-powered predictions through this dashboard."
)
st.divider()

# --- Key metrics ---
df_matches     = pd.read_parquet(PROCESSED / "matches.parquet")
df_players     = pd.read_parquet(PROCESSED / "players.parquet")
df_features    = pd.read_parquet(PROCESSED / "features.parquet")
df_tournaments = pd.read_parquet(PROCESSED / "tournaments.parquet")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Matches",     f"{len(df_matches):,}")
c2.metric("Labeled Matches",   f"{len(df_features):,}")
c3.metric("Players",           f"{len(df_players):,}")
c4.metric("Tournaments",       f"{len(df_tournaments):,}")
c5.metric("Model Accuracy",    "77.1%")

st.divider()

# --- Two columns: pipeline + results ---
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("How it works")
    st.markdown(
        """
        1. **Collect** — match, player, and tournament data fetched from the padelapi.org REST API
        2. **Engineer** — 33 pre-match features computed chronologically with no data leakage:
           ELO ratings, rolling win rate, form streak, pair chemistry, head-to-head history, days rest, ranking
        3. **Train** — XGBoost model trained on all labeled matches using a time-based split
        4. **Predict** — select any two pairs on the Match Predictor page to get a live win probability
        """
    )
    st.markdown("")
    st.subheader("Explore")
    st.markdown(
        """
        | Page | What you'll find |
        |---|---|
        | 🏆 ELO Rankings | Current standings computed from match history |
        | 🎯 Match Predictor | Pick 4 players, get a win probability |
        | 👤 Player Profile | ELO history, form chart, recent matches |
        | 📅 Tournament History | Browse tournaments and their results |
        | ⚔️ Head to Head | H2H record between two real pairs |
        | 📈 ELO Race | See how top players' ratings evolved over time |
        | 🤝 Pair Chemistry | Which pairs outperform their individual records |
        """
    )

with col_right:
    st.subheader("Model results")
    perf = pd.DataFrame({
        "Model":    ["ELO Baseline", "Logistic Regression", "Random Forest", "XGBoost (tuned)"],
        "Accuracy": ["64.2%", "74.3%", "75.4%", "77.1%"],
        "ROC-AUC":  ["—", "0.838", "0.831", "0.846"],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

    st.markdown("")
    st.subheader("Dataset split")
    cat_counts = (
        df_matches["category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Matches")
    )
    st.dataframe(cat_counts, use_container_width=True, hide_index=True)

    st.caption(
        "Free API tier hides match winners for ~83% of matches. "
        "891 labeled matches are available for training."
    )
