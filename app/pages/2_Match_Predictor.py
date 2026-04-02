import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import streamlit as st

from src.models.predict import get_elo_rankings, load_model, predict_match, PROCESSED
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Match Predictor", page_icon=":material/model_training:", layout="wide")
render_sidebar()
st.title("Match Predictor")
st.markdown("Select two pairs of players and get a predicted win probability from the trained XGBoost model.")

# --- Sidebar config ---
st.sidebar.header("Match Settings")
category = st.sidebar.radio("Category", ["men", "women"])
level = st.sidebar.selectbox(
    "Tournament Level",
    ["p1", "major", "wpt_master", "wpt_1000", "wpt_500", "p2", "fip_platinum", "fip_gold", "fip_other"],
)
round_ = st.sidebar.selectbox(
    "Round",
    options=[1, 2, 4, 8, 16, 32],
    format_func=lambda r: {1: "Final", 2: "Semi-Final", 4: "Quarter-Final",
                            8: "Round of 16", 16: "Round of 32", 32: "Round of 64"}[r],
    index=2,
)

# --- Player lookup ---
@st.cache_data
def load_players(cat):
    df = pd.read_parquet(PROCESSED / "players.parquet")
    return df[df["category"] == cat][["player_id", "name", "ranking"]].sort_values("ranking")

df_players = load_players(category)
player_options = {
    f"{row['name']} (#{int(row['ranking']) if pd.notna(row['ranking']) else '?'})": int(row["player_id"])
    for _, row in df_players.iterrows()
}
player_names = list(player_options.keys())

# --- Team selection ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Team 1")
    t1_p1_name = st.selectbox("Player 1", player_names, key="t1p1", index=0)
    t1_p2_name = st.selectbox("Player 2", player_names, key="t1p2", index=1)

with col2:
    st.subheader("Team 2")
    t2_p1_name = st.selectbox("Player 1", player_names, key="t2p1", index=2)
    t2_p2_name = st.selectbox("Player 2", player_names, key="t2p2", index=3)

t1 = [player_options[t1_p1_name], player_options[t1_p2_name]]
t2 = [player_options[t2_p1_name], player_options[t2_p2_name]]

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    if len(set(t1 + t2)) < 4:
        st.error("Each player must appear only once across both teams.")
    else:
        with st.spinner("Computing prediction..."):
            # Warm up model
            load_model()
            result = predict_match(t1, t2, category=category, level=level, round_=round_)

        prob_t1 = result["prob_t1"]
        prob_t2 = result["prob_t2"]

        st.subheader("Prediction")
        res_col1, res_col2 = st.columns(2)

        t1_label = f"{t1_p1_name.split(' (')[0]} / {t1_p2_name.split(' (')[0]}"
        t2_label = f"{t2_p1_name.split(' (')[0]} / {t2_p2_name.split(' (')[0]}"

        with res_col1:
            st.metric(f"Team 1 — {t1_label}", f"{prob_t1:.1%}")
            if prob_t1 > prob_t2:
                st.success("Predicted winner")

        with res_col2:
            st.metric(f"Team 2 — {t2_label}", f"{prob_t2:.1%}")
            if prob_t2 > prob_t1:
                st.success("Predicted winner")

        st.divider()
        st.subheader("Key Features Used")
        feat = result["features"]
        feat_df = pd.DataFrame({
            "Feature": ["ELO (T1)", "ELO (T2)", "ELO Diff", "Win Rate (T1)", "Win Rate (T2)",
                        "Ranking (T1)", "Ranking (T2)", "Form Streak (T1)", "Form Streak (T2)",
                        "Pair Matches (T1)", "Pair Matches (T2)", "H2H Total"],
            "Value": [
                round(feat["elo_t1"], 1), round(feat["elo_t2"], 1), round(feat["elo_diff"], 1),
                f"{feat['win_rate_t1']:.1%}", f"{feat['win_rate_t2']:.1%}",
                feat["ranking_t1"], feat["ranking_t2"],
                feat["form_streak_t1"], feat["form_streak_t2"],
                feat["pair_matches_t1"], feat["pair_matches_t2"], feat["h2h_total"],
            ],
        })
        st.dataframe(
            feat_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn(width="medium"),
                "Value":   st.column_config.TextColumn(width="small"),
            },
        )

        st.caption(
            "Model: XGBoost trained on 891 labeled matches (free API tier). "
            "Accuracy ~75% on held-out test set."
        )
