"""
Reusable prediction engine for match outcome.

Trains an XGBoost model on the full features dataset and exposes:
    - load_model()        → fitted pipeline (cached)
    - predict_match()     → win probability for a given pair vs pair
    - get_elo_rankings()  → current ELO standings from feature history
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.processing.features import (
    ELO_DEFAULT,
    TOURNAMENT_LEVEL_WEIGHTS,
    _expected,
    _rolling_win_rate,
    _form_streak,
    _days_since_last_match,
    _level_win_rate,
    _updated_elo,
)

PROCESSED = Path(__file__).parents[2] / "data" / "processed"

FEATURES = [
    "elo_t1", "elo_t2", "elo_diff",
    "win_rate_t1", "win_rate_t2", "win_rate_diff",
    "form_streak_t1", "form_streak_t2", "form_streak_diff",
    "days_rest_t1", "days_rest_t2", "days_rest_diff",
    "level_win_rate_t1", "level_win_rate_t2", "level_win_rate_diff",
    "pair_win_rate_t1", "pair_win_rate_t2", "pair_win_rate_diff",
    "pair_matches_t1", "pair_matches_t2",
    "matches_played_t1", "matches_played_t2", "matches_played_diff",
    "h2h_wins_t1", "h2h_wins_t2", "h2h_total", "h2h_win_rate_t1",
    "ranking_t1", "ranking_t2", "ranking_diff",
    "tournament_level_weight", "round", "category_men",
]


_BEST_PARAMS_PATH = Path(__file__).parent / "best_params.json"

_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _load_xgb_params() -> dict:
    """Load tuned params from best_params.json if available, else use defaults."""
    if _BEST_PARAMS_PATH.exists():
        with open(_BEST_PARAMS_PATH) as f:
            params = json.load(f)
        print(f"Using tuned params from {_BEST_PARAMS_PATH.name}")
        return params
    return _DEFAULT_PARAMS


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    """Train XGBoost on the full feature dataset and return the fitted pipeline."""
    df = pd.read_parquet(PROCESSED / "features.parquet")
    df["category_men"] = (df["category"] == "men").astype(int)

    X = df[FEATURES]
    y = df["target"]

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            **_load_xgb_params(),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )),
    ])
    pipe.fit(X, y)
    return pipe


@lru_cache(maxsize=1)
def _build_player_state() -> dict:
    """Replay all matches to build current ELO + history state for every player."""
    from collections import defaultdict

    df_matches = pd.read_parquet(PROCESSED / "matches.parquet")
    df_players = pd.read_parquet(PROCESSED / "players.parquet")
    df_tournaments = pd.read_parquet(PROCESSED / "tournaments.parquet")

    ranking_lookup: dict[int, float] = dict(
        zip(df_players["player_id"].astype(int), df_players["ranking"].astype(float))
    )
    level_lookup: dict[int, str] = dict(
        zip(df_tournaments["tournament_id"].astype(int), df_tournaments["level"].fillna("unknown"))
    )

    df = (
        df_matches
        .dropna(subset=["winner"])
        .query("winner in ('team_1', 'team_2')")
        .sort_values("played_at")
        .reset_index(drop=True)
    )

    elo: dict[int, float] = defaultdict(lambda: ELO_DEFAULT)
    player_history: dict[int, list] = defaultdict(list)
    pair_history: dict[frozenset, list] = defaultdict(list)
    level_history: dict[int, dict] = defaultdict(lambda: defaultdict(list))
    h2h: dict[tuple, dict] = defaultdict(lambda: {"t1": 0, "t2": 0})

    for _, match in df.iterrows():
        t1 = [int(p) for p in [match["t1_p1"], match["t1_p2"]] if pd.notna(p)]
        t2 = [int(p) for p in [match["t2_p1"], match["t2_p2"]] if pd.notna(p)]
        if not t1 or not t2:
            continue

        tid = match.get("tournament_id")
        level = level_lookup.get(int(tid), "unknown") if pd.notna(tid) else "unknown"
        t1_won = match["winner"] == "team_1"

        elo_t1 = float(np.mean([elo[p] for p in t1]))
        elo_t2 = float(np.mean([elo[p] for p in t2]))
        exp_t1 = _expected(elo_t1, elo_t2)
        new_elo_t1 = _updated_elo(elo_t1, exp_t1, 1.0 if t1_won else 0.0)
        new_elo_t2 = _updated_elo(elo_t2, 1.0 - exp_t1, 0.0 if t1_won else 1.0)

        for p in t1:
            elo[p] = new_elo_t1
            player_history[p].append((match["played_at"], t1_won))
            level_history[p][level].append(t1_won)
        for p in t2:
            elo[p] = new_elo_t2
            player_history[p].append((match["played_at"], not t1_won))
            level_history[p][level].append(not t1_won)

        pkey_t1, pkey_t2 = frozenset(t1), frozenset(t2)
        if len(t1) == 2:
            pair_history[pkey_t1].append((match["played_at"], t1_won))
        if len(t2) == 2:
            pair_history[pkey_t2].append((match["played_at"], not t1_won))

        h2h_key = (min(str(pkey_t1), str(pkey_t2)), max(str(pkey_t1), str(pkey_t2)))
        h2h_slot = "t1" if str(pkey_t1) <= str(pkey_t2) else "t2"
        if t1_won:
            h2h[h2h_key]["t1" if h2h_slot == "t1" else "t2"] += 1
        else:
            h2h[h2h_key]["t2" if h2h_slot == "t1" else "t1"] += 1

    return {
        "elo": dict(elo),
        "player_history": dict(player_history),
        "pair_history": {str(k): v for k, v in pair_history.items()},
        "level_history": dict(level_history),
        "h2h": dict(h2h),
        "ranking_lookup": ranking_lookup,
    }


def get_player_profile(player_id: int) -> dict:
    """Build a full profile for a player.

    Returns
    -------
    dict with keys:
        info          — player metadata row (Series)
        elo_history   — DataFrame(played_at, elo) after each match
        matches       — DataFrame of labeled matches with result, opponent names, tournament level
        win_rate      — overall win rate across labeled matches
        form_streak   — current consecutive W/L streak
    """
    df_matches = pd.read_parquet(PROCESSED / "matches.parquet")
    df_players = pd.read_parquet(PROCESSED / "players.parquet")
    df_tournaments = pd.read_parquet(PROCESSED / "tournaments.parquet")

    pid = int(player_id)
    info = df_players[df_players["player_id"] == pid].iloc[0]

    name_lookup: dict[int, str] = dict(
        zip(df_players["player_id"].astype(int), df_players["name"])
    )
    level_lookup: dict[int, str] = dict(
        zip(df_tournaments["tournament_id"].astype(int), df_tournaments["level"].fillna("unknown"))
    )

    df = (
        df_matches
        .dropna(subset=["winner"])
        .query("winner in ('team_1', 'team_2')")
        .sort_values("played_at")
        .reset_index(drop=True)
    )

    elo: dict[int, float] = defaultdict(lambda: ELO_DEFAULT)
    elo_history_rows = []
    match_rows = []

    for _, match in df.iterrows():
        t1 = [int(p) for p in [match["t1_p1"], match["t1_p2"]] if pd.notna(p)]
        t2 = [int(p) for p in [match["t2_p1"], match["t2_p2"]] if pd.notna(p)]
        if not t1 or not t2:
            continue

        t1_won = match["winner"] == "team_1"
        elo_t1 = float(np.mean([elo[p] for p in t1]))
        elo_t2 = float(np.mean([elo[p] for p in t2]))
        exp_t1 = _expected(elo_t1, elo_t2)
        new_elo_t1 = _updated_elo(elo_t1, exp_t1, 1.0 if t1_won else 0.0)
        new_elo_t2 = _updated_elo(elo_t2, 1.0 - exp_t1, 0.0 if t1_won else 1.0)

        tid = match.get("tournament_id")
        level = level_lookup.get(int(tid), "unknown") if pd.notna(tid) else "unknown"

        if pid in t1:
            won = t1_won
            partners = [p for p in t1 if p != pid]
            opponents = t2
            new_elo = new_elo_t1
        elif pid in t2:
            won = not t1_won
            partners = [p for p in t2 if p != pid]
            opponents = t1
            new_elo = new_elo_t2
        else:
            for p in t1:
                elo[p] = new_elo_t1
            for p in t2:
                elo[p] = new_elo_t2
            continue

        for p in t1:
            elo[p] = new_elo_t1
        for p in t2:
            elo[p] = new_elo_t2

        elo_history_rows.append({"played_at": match["played_at"], "elo": round(new_elo, 1)})

        partner_names = " / ".join(name_lookup.get(p, str(p)) for p in partners)
        opponent_names = " / ".join(name_lookup.get(p, str(p)) for p in opponents)
        match_rows.append({
            "date": match["played_at"],
            "result": "W" if won else "L",
            "partner": partner_names,
            "opponents": opponent_names,
            "level": level,
            "round": match.get("round_name", match.get("round")),
        })

    elo_history = pd.DataFrame(elo_history_rows)
    matches_df = pd.DataFrame(match_rows)

    state = _build_player_state()
    ph = state["player_history"].get(pid, [])
    win_rate = sum(1 for _, w in ph if w) / len(ph) if ph else None
    streak = _form_streak(ph)

    return {
        "info": info,
        "elo_history": elo_history,
        "matches": matches_df,
        "win_rate": win_rate,
        "form_streak": streak,
    }


def get_elo_rankings(category: str | None = None) -> pd.DataFrame:
    """Return a DataFrame of players ranked by current ELO.

    Parameters
    ----------
    category : 'men', 'women', or None for both
    """
    df_players = pd.read_parquet(PROCESSED / "players.parquet")
    state = _build_player_state()

    df_players["elo"] = df_players["player_id"].apply(
        lambda pid: state["elo"].get(int(pid), ELO_DEFAULT)
    )
    df_players["matches_played"] = df_players["player_id"].apply(
        lambda pid: len(state["player_history"].get(int(pid), []))
    )

    if category:
        df_players = df_players[df_players["category"] == category]

    return (
        df_players[["player_id", "name", "nationality", "category", "ranking", "elo", "matches_played"]]
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )


def predict_match(
    t1: list[int],
    t2: list[int],
    category: str = "men",
    level: str = "p1",
    round_: int = 4,
    window: int = 20,
) -> dict:
    """Predict win probability for team 1 vs team 2.

    Parameters
    ----------
    t1, t2   : lists of 1-2 player IDs
    category : 'men' or 'women'
    level    : tournament level string (e.g. 'p1', 'major')
    round_   : round number (1=Final, 2=Semi, 4=QF, ...)
    window   : rolling window for win-rate features

    Returns
    -------
    dict with keys: prob_t1, prob_t2, features
    """
    pipe = load_model()
    state = _build_player_state()

    elo = state["elo"]
    ph = state["player_history"]
    pair_h_raw = state["pair_history"]
    lh = state["level_history"]
    h2h = state["h2h"]
    ranking_lookup = state["ranking_lookup"]

    # pair_history was serialised with str keys
    pair_history = {frozenset(map(int, k.strip("frozenset({})").split(", "))): v
                    for k, v in pair_h_raw.items()}

    current_date = pd.Timestamp("now")

    elo_t1 = float(np.mean([elo.get(p, ELO_DEFAULT) for p in t1]))
    elo_t2 = float(np.mean([elo.get(p, ELO_DEFAULT) for p in t2]))

    wr_t1 = float(np.mean([_rolling_win_rate(ph.get(p, []), window) for p in t1]))
    wr_t2 = float(np.mean([_rolling_win_rate(ph.get(p, []), window) for p in t2]))

    streak_t1 = float(np.mean([_form_streak(ph.get(p, [])) for p in t1]))
    streak_t2 = float(np.mean([_form_streak(ph.get(p, [])) for p in t2]))

    rest_t1_vals = [_days_since_last_match(ph.get(p, []), current_date) for p in t1]
    rest_t2_vals = [_days_since_last_match(ph.get(p, []), current_date) for p in t2]
    rest_t1 = float(np.nanmean(rest_t1_vals)) if any(not np.isnan(v) for v in rest_t1_vals) else np.nan
    rest_t2 = float(np.nanmean(rest_t2_vals)) if any(not np.isnan(v) for v in rest_t2_vals) else np.nan

    level_weight = TOURNAMENT_LEVEL_WEIGHTS.get(level, 1)
    lvl_wr_t1 = float(np.mean([_level_win_rate(lh.get(p, {}), level, window) for p in t1]))
    lvl_wr_t2 = float(np.mean([_level_win_rate(lh.get(p, {}), level, window) for p in t2]))

    pkey_t1, pkey_t2 = frozenset(t1), frozenset(t2)
    pair_wr_t1 = _rolling_win_rate(pair_history.get(pkey_t1, []), window) if len(t1) == 2 else wr_t1
    pair_wr_t2 = _rolling_win_rate(pair_history.get(pkey_t2, []), window) if len(t2) == 2 else wr_t2
    pair_matches_t1 = len(pair_history.get(pkey_t1, []))
    pair_matches_t2 = len(pair_history.get(pkey_t2, []))

    exp_t1 = float(np.mean([len(ph.get(p, [])) for p in t1]))
    exp_t2 = float(np.mean([len(ph.get(p, [])) for p in t2]))

    h2h_key = (min(str(pkey_t1), str(pkey_t2)), max(str(pkey_t1), str(pkey_t2)))
    h2h_slot = "t1" if str(pkey_t1) <= str(pkey_t2) else "t2"
    h2h_rec = h2h.get(h2h_key, {"t1": 0, "t2": 0})
    h2h_wins_t1 = h2h_rec["t1"] if h2h_slot == "t1" else h2h_rec["t2"]
    h2h_wins_t2 = h2h_rec["t2"] if h2h_slot == "t1" else h2h_rec["t1"]
    h2h_total = h2h_wins_t1 + h2h_wins_t2
    h2h_wr_t1 = h2h_wins_t1 / h2h_total if h2h_total > 0 else 0.5

    rank_t1 = float(np.nanmean([ranking_lookup.get(p, np.nan) for p in t1]))
    rank_t2 = float(np.nanmean([ranking_lookup.get(p, np.nan) for p in t2]))
    rank_diff = (rank_t1 - rank_t2) if not (np.isnan(rank_t1) or np.isnan(rank_t2)) else np.nan

    feat = {
        "elo_t1": elo_t1, "elo_t2": elo_t2, "elo_diff": elo_t1 - elo_t2,
        "win_rate_t1": wr_t1, "win_rate_t2": wr_t2, "win_rate_diff": wr_t1 - wr_t2,
        "form_streak_t1": streak_t1, "form_streak_t2": streak_t2, "form_streak_diff": streak_t1 - streak_t2,
        "days_rest_t1": rest_t1, "days_rest_t2": rest_t2,
        "days_rest_diff": (rest_t1 - rest_t2) if not (np.isnan(rest_t1) or np.isnan(rest_t2)) else np.nan,
        "level_win_rate_t1": lvl_wr_t1, "level_win_rate_t2": lvl_wr_t2, "level_win_rate_diff": lvl_wr_t1 - lvl_wr_t2,
        "pair_win_rate_t1": pair_wr_t1, "pair_win_rate_t2": pair_wr_t2, "pair_win_rate_diff": pair_wr_t1 - pair_wr_t2,
        "pair_matches_t1": pair_matches_t1, "pair_matches_t2": pair_matches_t2,
        "matches_played_t1": exp_t1, "matches_played_t2": exp_t2, "matches_played_diff": exp_t1 - exp_t2,
        "h2h_wins_t1": h2h_wins_t1, "h2h_wins_t2": h2h_wins_t2, "h2h_total": h2h_total, "h2h_win_rate_t1": h2h_wr_t1,
        "ranking_t1": rank_t1, "ranking_t2": rank_t2, "ranking_diff": rank_diff,
        "tournament_level_weight": level_weight, "round": round_,
        "category_men": 1 if category == "men" else 0,
    }

    X = pd.DataFrame([feat])[FEATURES]
    prob_t1 = float(pipe.predict_proba(X)[0, 1])

    return {"prob_t1": prob_t1, "prob_t2": 1 - prob_t1, "features": feat}
