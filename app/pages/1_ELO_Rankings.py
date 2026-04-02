import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import plotly.express as px
import streamlit as st

from src.models.predict import get_elo_rankings

st.set_page_config(page_title="ELO Rankings", page_icon="🎾", layout="wide")
st.title("ELO Rankings")
st.markdown("Current ELO ratings computed from all labeled matches, updated after every result.")

category = st.radio("Category", ["men", "women"], horizontal=True)

with st.spinner("Building ELO state from match history..."):
    df = get_elo_rankings(category=category)

df.index = df.index + 1  # 1-based rank
df_display = df[["name", "nationality", "elo", "ranking", "matches_played"]].copy()
df_display.columns = ["Player", "Nationality", "ELO", "Official Ranking", "Matches Played"]
df_display["ELO"] = df_display["ELO"].round(1)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Leaderboard")
    st.dataframe(df_display.head(30), use_container_width=True)

with col2:
    st.subheader("ELO vs Official Ranking (Top 50)")
    top50 = df.head(50).copy()
    top50["elo_rank"] = range(1, len(top50) + 1)
    fig = px.scatter(
        top50,
        x="ranking",
        y="elo",
        hover_name="name",
        hover_data={"nationality": True, "matches_played": True},
        labels={"ranking": "Official Ranking", "elo": "ELO Rating"},
        title=f"ELO vs Official Ranking — {category}",
    )
    fig.update_traces(marker=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("ELO Distribution")
fig2 = px.histogram(
    df[df["matches_played"] >= 5],
    x="elo",
    nbins=40,
    labels={"elo": "ELO Rating", "count": "Players"},
    title=f"ELO distribution — {category} (min 5 matches)",
)
st.plotly_chart(fig2, use_container_width=True)
