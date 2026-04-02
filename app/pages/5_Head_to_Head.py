import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import streamlit as st

from src.models.analysis import get_real_pairs, get_pairs_faced, get_h2h_record
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Head to Head", page_icon=":material/compare_arrows:", layout="wide")
render_sidebar()
st.title("Head to Head")
st.markdown("Compare two pairs that have actually faced each other.")

category = st.radio("Category", ["men", "women"], horizontal=True)

with st.spinner("Loading pairs..."):
    all_pairs = get_real_pairs()
    cat_pairs = all_pairs[all_pairs["category"] == category].copy()

if cat_pairs.empty:
    st.warning("No pairs found for this category.")
    st.stop()

# --- Pair 1 ---
pair1_options = {row["names"]: row["pair_key"] for _, row in cat_pairs.iterrows()}
pair1_name    = st.selectbox("Pair 1", list(pair1_options.keys()))
pair1_ids     = pair1_options[pair1_name]

# --- Pair 2: only pairs that have faced Pair 1 ---
faced_keys = get_pairs_faced(pair1_ids)
faced_keys = [k for k in faced_keys if set(k) != set(pair1_ids)]

# Filter to pairs that are in our real pairs list (have min_matches together)
real_keys  = set(cat_pairs["pair_key"].tolist())
faced_keys = [k for k in faced_keys if k in real_keys]

if not faced_keys:
    st.info(f"No pairs with sufficient match history have faced **{pair1_name}** in labeled matches.")
    st.stop()

name_map   = dict(zip(cat_pairs["pair_key"], cat_pairs["names"]))
pair2_options = {name_map[k]: k for k in faced_keys if k in name_map}

if not pair2_options:
    st.info("No eligible opponents found.")
    st.stop()

pair2_name = st.selectbox("Pair 2", list(pair2_options.keys()))
pair2_ids  = pair2_options[pair2_name]

st.divider()

with st.spinner("Loading H2H record..."):
    h2h = get_h2h_record(pair1_ids, pair2_ids)

if h2h["total"] == 0:
    st.info("These pairs have no head-to-head matches in the labeled dataset.")
    st.stop()

# --- Summary ---
c1, c2, c3 = st.columns(3)
c1.metric(f"{h2h['pair1_names']}", f"{h2h['wins1']} wins")
c2.metric("Total meetings", h2h["total"])
c3.metric(f"{h2h['pair2_names']}", f"{h2h['wins2']} wins")

# Win rate bar
wr1 = h2h["wins1"] / h2h["total"]
st.markdown(f"**Win rate:** {h2h['pair1_names']} — {wr1:.0%} &nbsp;|&nbsp; {h2h['pair2_names']} — {1 - wr1:.0%}")
st.progress(wr1)

st.divider()

# --- Match history ---
st.subheader("Match history")
matches = h2h["matches"]
if len(matches):
    matches["date"] = matches["date"].dt.date
    matches.columns = ["Date", "Winner", "Round", "Tournament", "Level"]
    st.dataframe(
        matches,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date":       st.column_config.DateColumn(width="small"),
            "Winner":     st.column_config.TextColumn(width="large"),
            "Round":      st.column_config.TextColumn(width="small"),
            "Tournament": st.column_config.TextColumn(width="large"),
            "Level":      st.column_config.TextColumn(width="small"),
        },
    )
