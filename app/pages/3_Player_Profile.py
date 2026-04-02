import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.predict import get_player_profile, PROCESSED
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Player Profile", page_icon=":material/person:", layout="wide")
render_sidebar()
st.title("Player Profile")

# --- Player selector ---
@st.cache_data
def load_players():
    df = pd.read_parquet(PROCESSED / "players.parquet")
    return df[["player_id", "name", "category", "ranking"]].sort_values("ranking")

df_players = load_players()

category = st.radio("Category", ["men", "women"], horizontal=True)
df_cat = df_players[df_players["category"] == category]

player_options = {
    f"{row['name']} (#{int(row['ranking']) if pd.notna(row['ranking']) else '?'})": int(row["player_id"])
    for _, row in df_cat.iterrows()
}

selected_name = st.selectbox("Select player", list(player_options.keys()))
player_id = player_options[selected_name]

with st.spinner("Loading profile..."):
    profile = get_player_profile(player_id)

info        = profile["info"]
elo_history = profile["elo_history"]
matches     = profile["matches"]
win_rate    = profile["win_rate"]
streak      = profile["form_streak"]

# --- Header ---
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Official Ranking", f"#{int(info['ranking'])}" if pd.notna(info["ranking"]) else "—")
col2.metric("Current ELO", f"{elo_history['elo'].iloc[-1]:.0f}" if len(elo_history) else "—")
col3.metric("Win Rate", f"{win_rate:.1%}" if win_rate is not None else "—")
col4.metric("Matches Played", len(matches))
streak_label = f"+{streak} W streak" if streak > 0 else (f"{streak} L streak" if streak < 0 else "—")
col5.metric("Form Streak", streak_label)

nationality = info.get("nationality", "")
hand = info.get("hand", "")
side = info.get("side", "")
details = " · ".join(x for x in [nationality, hand, side] if x)
if details:
    st.caption(details)

st.divider()

# --- ELO history chart ---
if len(elo_history) >= 2:
    st.subheader("ELO over time")
    fig = px.line(
        elo_history,
        x="played_at",
        y="elo",
        labels={"played_at": "Date", "elo": "ELO Rating"},
    )
    fig.add_hline(y=1500, line_dash="dot", line_color="grey", annotation_text="Starting ELO")
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough labeled matches to plot ELO history.")

# --- Win rate by tournament level ---
if len(matches) > 0:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Win rate by tournament level")
        level_stats = (
            matches.groupby("level")["result"]
            .apply(lambda x: (x == "W").sum() / len(x))
            .reset_index()
        )
        level_stats.columns = ["Level", "Win Rate"]
        level_stats = level_stats.sort_values("Win Rate", ascending=True)
        fig2 = px.bar(
            level_stats,
            x="Win Rate",
            y="Level",
            orientation="h",
            labels={"Win Rate": "Win Rate"},
        )
        fig2.update_xaxes(tickformat=".0%", range=[0, 1])
        fig2.add_vline(x=0.5, line_dash="dot", line_color="grey")
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.subheader("W/L over time (rolling 10)")
        matches_sorted = matches.sort_values("date").copy()
        matches_sorted["won"] = (matches_sorted["result"] == "W").astype(int)
        matches_sorted["rolling_wr"] = matches_sorted["won"].rolling(10, min_periods=3).mean()
        fig3 = px.line(
            matches_sorted.dropna(subset=["rolling_wr"]),
            x="date",
            y="rolling_wr",
            labels={"date": "Date", "rolling_wr": "Win Rate (rolling 10)"},
        )
        fig3.update_yaxes(tickformat=".0%", range=[0, 1])
        fig3.add_hline(y=0.5, line_dash="dot", line_color="grey")
        st.plotly_chart(fig3, use_container_width=True)

# --- Recent matches ---
st.subheader("Recent matches")
if len(matches) > 0:
    recent = matches.sort_values("date", ascending=False).head(15).copy()
    recent["date"] = recent["date"].dt.date
    recent.columns = ["Date", "Result", "Partner", "Opponents", "Level", "Round"]
    st.dataframe(
        recent.style.map(
            lambda v: "color: green; font-weight: bold" if v == "W" else "color: red; font-weight: bold",
            subset=["Result"],
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date":      st.column_config.DateColumn(width="small"),
            "Result":    st.column_config.TextColumn(width="small"),
            "Partner":   st.column_config.TextColumn(width="large"),
            "Opponents": st.column_config.TextColumn(width="large"),
            "Level":     st.column_config.TextColumn(width="small"),
            "Round":     st.column_config.TextColumn(width="small"),
        },
    )
else:
    st.info("No labeled matches found for this player.")
