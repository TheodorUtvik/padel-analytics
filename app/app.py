"""
Padel Analytics — Streamlit Dashboard
Run with: streamlit run app/app.py
"""
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import streamlit as st

PROCESSED = Path(__file__).parents[1] / "data" / "processed"

st.set_page_config(
    page_title="Padel Analytics",
    page_icon="🎾",
    layout="wide",
)

st.title("Padel Analytics")
st.markdown("Professional padel match data — feature engineering, ELO rankings, and match outcome prediction.")

# --- Quick stats ---
df_matches     = pd.read_parquet(PROCESSED / "matches.parquet")
df_players     = pd.read_parquet(PROCESSED / "players.parquet")
df_features    = pd.read_parquet(PROCESSED / "features.parquet")
df_tournaments = pd.read_parquet(PROCESSED / "tournaments.parquet")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Matches", f"{len(df_matches):,}")
col2.metric("Labeled Matches", f"{len(df_features):,}")
col3.metric("Players", f"{len(df_players):,}")
col4.metric("Tournaments", f"{len(df_tournaments):,}")

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Matches by Category")
    cat_counts = df_matches["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Matches"]
    st.dataframe(cat_counts, use_container_width=True, hide_index=True)

with col_b:
    st.subheader("Model Performance")
    perf = pd.DataFrame({
        "Model": ["ELO Baseline", "Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": ["64.2%", "74.3%", "75.4%", "75.4%"],
        "ROC-AUC": ["—", "0.838", "0.831", "0.838"],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

st.divider()
st.markdown("**Navigate** using the sidebar to explore ELO rankings or predict a match.")
