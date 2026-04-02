import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import plotly.express as px
import streamlit as st

from src.models.analysis import get_elo_timeline
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="ELO Race", page_icon="📈", layout="wide")
render_sidebar()
st.title("📈 ELO Race")
st.markdown("How the top players' ELO ratings evolved over time.")

category = st.radio("Category", ["men", "women"], horizontal=True)
top_n    = st.slider("Number of players to show", min_value=3, max_value=15, value=8)

with st.spinner("Building ELO timeline..."):
    df = get_elo_timeline()

df_cat = df[df["category"] == category]

# Identify top N players by their final ELO
final_elo  = df_cat.groupby("player_id")["elo"].last()
top_ids    = final_elo.nlargest(top_n).index.tolist()
df_top     = df_cat[df_cat["player_id"].isin(top_ids)].copy()

# Keep only players with enough matches to make the chart readable
df_top = df_top[df_top["player_id"].isin(
    df_top.groupby("player_id").size()[lambda x: x >= 3].index
)]

fig = px.line(
    df_top,
    x="played_at",
    y="elo",
    color="name",
    labels={"played_at": "Date", "elo": "ELO Rating", "name": "Player"},
    title=f"ELO Race — Top {top_n} {category} players",
)
fig.add_hline(y=1500, line_dash="dot", line_color="grey",
              annotation_text="Starting ELO", annotation_position="bottom right")
fig.update_traces(line=dict(width=2))
fig.update_layout(legend=dict(orientation="v", x=1.01, y=1))
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader(f"Current top {top_n} ELO — {category}")
leaderboard = (
    final_elo[top_ids]
    .reset_index()
    .merge(df_cat[["player_id", "name"]].drop_duplicates(), on="player_id")
    [["name", "elo"]]
    .sort_values("elo", ascending=False)
    .reset_index(drop=True)
)
leaderboard.index += 1
leaderboard.columns = ["Player", "ELO"]
leaderboard["ELO"] = leaderboard["ELO"].round(1)
st.dataframe(leaderboard, use_container_width=True)
