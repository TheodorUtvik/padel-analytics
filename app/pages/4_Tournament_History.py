import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import streamlit as st

from src.models.analysis import get_tournament_history, PROCESSED
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Tournament History", page_icon=":material/calendar_month:", layout="wide")
render_sidebar()
st.title("Tournament History")

with st.spinner("Loading tournaments..."):
    df = get_tournament_history()

category = st.radio("Category", ["men", "women", "both"], horizontal=True)
if category != "both":
    df = df[df["category"] == category]

# --- Overview table ---
st.subheader(f"All tournaments ({len(df)})")

display = df[["name", "location", "country", "level", "start_date", "matches", "winner"]].copy()
display.columns = ["Tournament", "Location", "Country", "Level", "Date", "Matches", "Winner"]
display["Date"] = pd.to_datetime(display["Date"]).dt.date

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Tournament": st.column_config.TextColumn(width="large"),
        "Winner":     st.column_config.TextColumn(width="large"),
        "Location":   st.column_config.TextColumn(width="medium"),
        "Country":    st.column_config.TextColumn(width="small"),
        "Level":      st.column_config.TextColumn(width="small"),
        "Matches":    st.column_config.NumberColumn(width="small"),
        "Date":       st.column_config.DateColumn(width="small"),
    },
)

st.divider()

# --- Tournament drill-down ---
st.subheader("Tournament detail")
tournament_names = df["name"].dropna().unique().tolist()
selected = st.selectbox("Select tournament", sorted(tournament_names))

t_row = df[df["name"] == selected].iloc[0]
tid   = int(t_row["tournament_id"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Level",    t_row["level"] or "—")
c2.metric("Location", t_row["location"] or "—")
c3.metric("Matches",  t_row["matches"])
c4.metric("Winner",   t_row["winner"])

# Show all matches in that tournament
df_matches  = pd.read_parquet(PROCESSED / "matches.parquet")
df_players  = pd.read_parquet(PROCESSED / "players.parquet")
name_lookup = dict(zip(df_players["player_id"].astype(int), df_players["name"]))

t_matches = df_matches[
    (df_matches["tournament_id"] == tid) &
    (df_matches["category"] == t_row["category"])
].copy()

if len(t_matches):
    def pair_names(row, side):
        ids = [row[f"{side}_p1"], row[f"{side}_p2"]]
        return " / ".join(name_lookup.get(int(p), "?") for p in ids if pd.notna(p))

    t_matches["Team 1"] = t_matches.apply(lambda r: pair_names(r, "t1"), axis=1)
    t_matches["Team 2"] = t_matches.apply(lambda r: pair_names(r, "t2"), axis=1)
    t_matches["Result"] = t_matches["winner"].map(
        {"team_1": "Team 1 won", "team_2": "Team 2 won", "hidden_free_plan": "Hidden"}
    )

    show = t_matches[["round_name", "Team 1", "Team 2", "Result"]].copy()
    show.columns = ["Round", "Team 1", "Team 2", "Result"]
    show = show.sort_values("Round")

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Round":  st.column_config.TextColumn(width="small"),
            "Team 1": st.column_config.TextColumn(width="large"),
            "Team 2": st.column_config.TextColumn(width="large"),
            "Result": st.column_config.TextColumn(width="small"),
        },
    )
else:
    st.info("No match detail available for this tournament.")
