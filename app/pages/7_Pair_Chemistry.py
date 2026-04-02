import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import plotly.express as px
import streamlit as st

from src.models.analysis import get_pair_chemistry
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Pair Chemistry", page_icon=":material/group:", layout="wide")
render_sidebar()
st.title("Pair Chemistry")
st.markdown(
    "Which pairs perform **better together** than their individual records suggest? "
    "Chemistry = pair win rate − average individual win rate. "
    "Positive means the pair is more than the sum of its parts."
)

category = st.radio("Category", ["men", "women"], horizontal=True)
min_matches = st.slider("Minimum matches together", min_value=3, max_value=15, value=5)

with st.spinner("Computing pair chemistry..."):
    df = get_pair_chemistry(min_matches=min_matches)

df_cat = df[df["category"] == category].copy()

if df_cat.empty:
    st.warning("No pairs found with enough matches. Try lowering the minimum.")
    st.stop()

top_n = min(20, len(df_cat))
df_top = df_cat.head(top_n)
df_bot = df_cat.tail(top_n).sort_values("chemistry")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Top {top_n} — best chemistry")
    fig1 = px.bar(
        df_top.sort_values("chemistry"),
        x="chemistry",
        y="pair",
        orientation="h",
        color="chemistry",
        color_continuous_scale=["#d73027", "#fee090", "#4575b4"],
        color_continuous_midpoint=0,
        labels={"chemistry": "Chemistry Score", "pair": ""},
    )
    fig1.update_layout(coloraxis_showscale=False)
    fig1.add_vline(x=0, line_dash="dot", line_color="grey")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader(f"Bottom {top_n} — worst chemistry")
    fig2 = px.bar(
        df_bot,
        x="chemistry",
        y="pair",
        orientation="h",
        color="chemistry",
        color_continuous_scale=["#d73027", "#fee090", "#4575b4"],
        color_continuous_midpoint=0,
        labels={"chemistry": "Chemistry Score", "pair": ""},
    )
    fig2.update_layout(coloraxis_showscale=False)
    fig2.add_vline(x=0, line_dash="dot", line_color="grey")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Full table")
display = df_cat[["pair", "matches_together", "pair_win_rate", "avg_individual_win_rate", "chemistry"]].copy()
display.columns = ["Pair", "Matches Together", "Pair Win Rate", "Avg Individual Win Rate", "Chemistry"]
display["Pair Win Rate"]            = display["Pair Win Rate"].map("{:.1%}".format)
display["Avg Individual Win Rate"]  = display["Avg Individual Win Rate"].map("{:.1%}".format)
display["Chemistry"]                = display["Chemistry"].map("{:+.1%}".format)
st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Pair":                      st.column_config.TextColumn(width="large"),
        "Matches Together":          st.column_config.NumberColumn(width="small"),
        "Pair Win Rate":             st.column_config.TextColumn(width="small"),
        "Avg Individual Win Rate":   st.column_config.TextColumn(width="small"),
        "Chemistry":                 st.column_config.TextColumn(width="small"),
    },
)
