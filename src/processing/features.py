"""
Feature engineering for match outcome prediction.

All features are computed using ONLY information available before each match —
no future data is ever used (no leakage). Matches are processed in
chronological order; state (ELO, histories) is updated AFTER each row is
recorded.

Main entry point:
    df_features = compute_features(df_matches, df_players, df_tournaments)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ELO constants
# ---------------------------------------------------------------------------
ELO_DEFAULT = 1500.0
ELO_K = 32.0

# Tournament prestige weights (used as a feature, not for ELO weighting)
TOURNAMENT_LEVEL_WEIGHTS: dict[str, int] = {
    "major": 5,
    "wpt_master": 5,
    "wpt_final": 5,
    "wpt_1000": 4,
    "p1": 4,
    "wpt_500": 3,
    "p2": 3,
    "fip_platinum": 2,
    "fip_gold": 2,
    "finals": 3,
    "fip_other": 1,
}


# ---------------------------------------------------------------------------
# ELO helpers
# ---------------------------------------------------------------------------

def _expected(rating_a: float, rating_b: float) -> float:
    """ELO expected score for A against B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _updated_elo(rating: float, expected: float, actual: float) -> float:
    return rating + ELO_K * (actual - expected)


# ---------------------------------------------------------------------------
# Rolling / history helpers
# ---------------------------------------------------------------------------

def _rolling_win_rate(history: list[tuple], window: int) -> float:
    """Win rate over the last `window` entries in history list [(date, won)]."""
    recent = history[-window:]
    if not recent:
        return 0.5
    return sum(1 for _, won in recent if won) / len(recent)


def _form_streak(history: list[tuple]) -> int:
    """Current consecutive win/loss streak.

    Positive = win streak, negative = loss streak.
    e.g. W,W,W → +3  |  L,L → -2  |  empty → 0
    """
    if not history:
        return 0
    streak = 0
    current = history[-1][1]   # True = win, False = loss
    for _, won in reversed(history):
        if won == current:
            streak += 1 if won else -1
        else:
            break
    return streak


def _days_since_last_match(history: list[tuple], current_date: pd.Timestamp) -> float:
    """Days between the player's last match and the current match date.

    Returns NaN if the player has no prior matches.
    """
    if not history:
        return np.nan
    last_date = pd.Timestamp(history[-1][0])
    delta = (current_date - last_date).days
    return float(max(delta, 0))


def _level_win_rate(level_history: dict[str, list[bool]], level: str, window: int) -> float:
    """Win rate at a specific tournament level over the last `window` matches at that level."""
    hist = level_history.get(level, [])[-window:]
    if not hist:
        return 0.5
    return sum(hist) / len(hist)


def _avg_ranking(player_ids: list[int], ranking_lookup: dict[int, float]) -> float:
    ranks = [ranking_lookup.get(pid, np.nan) for pid in player_ids]
    valid = [r for r in ranks if not np.isnan(r)]
    return float(np.mean(valid)) if valid else np.nan


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(
    df_matches: pd.DataFrame,
    df_players: pd.DataFrame,
    df_tournaments: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Compute pre-match features for every finished match with a known winner.

    Parameters
    ----------
    df_matches : DataFrame
        Output of flatten_match — must contain columns:
        match_id, tournament_id, category, round, played_at, winner,
        t1_p1, t1_p2, t2_p1, t2_p2.
    df_players : DataFrame
        Output of flatten_player — must contain player_id, ranking.
    df_tournaments : DataFrame
        Output of flatten_tournament — must contain tournament_id, level.
    window : int
        Rolling window size (number of past matches) for win-rate features.

    Returns
    -------
    DataFrame with one row per match and all engineered features + target.

    Feature groups
    --------------
    ELO             — elo_t1, elo_t2, elo_diff
    Win rate        — win_rate_t1/t2/diff  (last `window` matches)
    Form streak     — form_streak_t1/t2/diff  (consecutive W/L run)
    Pair chemistry  — pair_win_rate_t1/t2/diff, pair_matches_t1/t2
    Level win rate  — level_win_rate_t1/t2/diff  (at this tournament tier)
    Rest            — days_rest_t1/t2/diff  (days since last match)
    Experience      — matches_played_t1/t2/diff
    H2H             — h2h_wins_t1/t2, h2h_total, h2h_win_rate_t1
    Ranking         — ranking_t1/t2/diff
    Context         — tournament_level_weight, round, category
    """
    # Sort chronologically — required for leak-free computation
    df = (
        df_matches
        .dropna(subset=["winner"])
        .query("winner in ('team_1', 'team_2')")
        .sort_values("played_at")
        .reset_index(drop=True)
    )

    # --- Lookup tables ---
    ranking_lookup: dict[int, float] = dict(
        zip(df_players["player_id"].astype(int), df_players["ranking"].astype(float))
    )
    level_lookup: dict[int, str] = dict(
        zip(df_tournaments["tournament_id"].astype(int), df_tournaments["level"].fillna("unknown"))
    )

    # --- Mutable state (updated AFTER each match) ---
    elo: dict[int, float] = defaultdict(lambda: ELO_DEFAULT)
    player_history: dict[int, list[tuple]] = defaultdict(list)    # [(played_at, won)]
    pair_history: dict[frozenset, list[tuple]] = defaultdict(list)
    level_history: dict[int, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    h2h: dict[tuple, dict[str, int]] = defaultdict(lambda: {"t1": 0, "t2": 0})

    rows = []

    for _, match in df.iterrows():
        t1 = [int(p) for p in [match["t1_p1"], match["t1_p2"]] if pd.notna(p)]
        t2 = [int(p) for p in [match["t2_p1"], match["t2_p2"]] if pd.notna(p)]

        if not t1 or not t2:
            continue

        current_date = pd.Timestamp(match["played_at"])

        # ── Pre-match features ──────────────────────────────────────────────

        # ELO
        elo_t1 = float(np.mean([elo[p] for p in t1]))
        elo_t2 = float(np.mean([elo[p] for p in t2]))

        # Rolling win rate
        wr_t1 = float(np.mean([_rolling_win_rate(player_history[p], window) for p in t1]))
        wr_t2 = float(np.mean([_rolling_win_rate(player_history[p], window) for p in t2]))

        # Form streak (avg over pair members)
        streak_t1 = float(np.mean([_form_streak(player_history[p]) for p in t1]))
        streak_t2 = float(np.mean([_form_streak(player_history[p]) for p in t2]))

        # Days since last match (avg over pair members)
        rest_t1_vals = [_days_since_last_match(player_history[p], current_date) for p in t1]
        rest_t2_vals = [_days_since_last_match(player_history[p], current_date) for p in t2]
        rest_t1 = float(np.nanmean(rest_t1_vals)) if any(not np.isnan(v) for v in rest_t1_vals) else np.nan
        rest_t2 = float(np.nanmean(rest_t2_vals)) if any(not np.isnan(v) for v in rest_t2_vals) else np.nan

        # Tournament level
        tid = match.get("tournament_id")
        level = level_lookup.get(int(tid), "unknown") if pd.notna(tid) else "unknown"
        level_weight = TOURNAMENT_LEVEL_WEIGHTS.get(level, 1)

        # Win rate at this specific tournament level
        lvl_wr_t1 = float(np.mean([_level_win_rate(level_history[p], level, window) for p in t1]))
        lvl_wr_t2 = float(np.mean([_level_win_rate(level_history[p], level, window) for p in t2]))

        # Pair win rate
        pkey_t1 = frozenset(t1)
        pkey_t2 = frozenset(t2)
        pair_wr_t1 = _rolling_win_rate(pair_history[pkey_t1], window) if len(t1) == 2 else wr_t1
        pair_wr_t2 = _rolling_win_rate(pair_history[pkey_t2], window) if len(t2) == 2 else wr_t2
        pair_matches_t1 = len(pair_history[pkey_t1])
        pair_matches_t2 = len(pair_history[pkey_t2])

        # Experience
        exp_t1 = float(np.mean([len(player_history[p]) for p in t1]))
        exp_t2 = float(np.mean([len(player_history[p]) for p in t2]))

        # H2H between these two exact pairs
        h2h_key = (min(str(pkey_t1), str(pkey_t2)), max(str(pkey_t1), str(pkey_t2)))
        h2h_slot = "t1" if str(pkey_t1) <= str(pkey_t2) else "t2"
        h2h_rec = h2h[h2h_key]
        h2h_wins_t1 = h2h_rec["t1"] if h2h_slot == "t1" else h2h_rec["t2"]
        h2h_wins_t2 = h2h_rec["t2"] if h2h_slot == "t1" else h2h_rec["t1"]
        h2h_total = h2h_wins_t1 + h2h_wins_t2
        h2h_wr_t1 = h2h_wins_t1 / h2h_total if h2h_total > 0 else 0.5

        # Ranking
        rank_t1 = _avg_ranking(t1, ranking_lookup)
        rank_t2 = _avg_ranking(t2, ranking_lookup)
        rank_diff = (rank_t1 - rank_t2) if not (np.isnan(rank_t1) or np.isnan(rank_t2)) else np.nan

        # Target
        t1_won = match["winner"] == "team_1"

        rows.append({
            # Identifiers
            "match_id":                 match["match_id"],
            "played_at":                match["played_at"],
            "category":                 match["category"],
            "round":                    match["round"],
            "tournament_id":            tid,
            "tournament_level":         level,
            "tournament_level_weight":  level_weight,
            # ELO
            "elo_t1":                   round(elo_t1, 2),
            "elo_t2":                   round(elo_t2, 2),
            "elo_diff":                 round(elo_t1 - elo_t2, 2),
            # Rolling win rate
            "win_rate_t1":              round(wr_t1, 4),
            "win_rate_t2":              round(wr_t2, 4),
            "win_rate_diff":            round(wr_t1 - wr_t2, 4),
            # Form streak
            "form_streak_t1":           round(streak_t1, 2),
            "form_streak_t2":           round(streak_t2, 2),
            "form_streak_diff":         round(streak_t1 - streak_t2, 2),
            # Days rest
            "days_rest_t1":             round(rest_t1, 1) if not np.isnan(rest_t1) else np.nan,
            "days_rest_t2":             round(rest_t2, 1) if not np.isnan(rest_t2) else np.nan,
            "days_rest_diff":           round(rest_t1 - rest_t2, 1) if not (np.isnan(rest_t1) or np.isnan(rest_t2)) else np.nan,
            # Level win rate
            "level_win_rate_t1":        round(lvl_wr_t1, 4),
            "level_win_rate_t2":        round(lvl_wr_t2, 4),
            "level_win_rate_diff":      round(lvl_wr_t1 - lvl_wr_t2, 4),
            # Pair chemistry
            "pair_win_rate_t1":         round(pair_wr_t1, 4),
            "pair_win_rate_t2":         round(pair_wr_t2, 4),
            "pair_win_rate_diff":       round(pair_wr_t1 - pair_wr_t2, 4),
            "pair_matches_t1":          pair_matches_t1,
            "pair_matches_t2":          pair_matches_t2,
            # Experience
            "matches_played_t1":        exp_t1,
            "matches_played_t2":        exp_t2,
            "matches_played_diff":      exp_t1 - exp_t2,
            # H2H
            "h2h_wins_t1":              h2h_wins_t1,
            "h2h_wins_t2":              h2h_wins_t2,
            "h2h_total":                h2h_total,
            "h2h_win_rate_t1":          round(h2h_wr_t1, 4),
            # Ranking
            "ranking_t1":               rank_t1,
            "ranking_t2":               rank_t2,
            "ranking_diff":             rank_diff,
            # Target
            "target":                   int(t1_won),
        })

        # ── Update state AFTER recording features ───────────────────────────

        exp_elo_t1 = _expected(elo_t1, elo_t2)
        new_elo_t1 = _updated_elo(elo_t1, exp_elo_t1, 1.0 if t1_won else 0.0)
        new_elo_t2 = _updated_elo(elo_t2, 1.0 - exp_elo_t1, 0.0 if t1_won else 1.0)

        for p in t1:
            elo[p] = new_elo_t1
            player_history[p].append((match["played_at"], t1_won))
            level_history[p][level].append(t1_won)
        for p in t2:
            elo[p] = new_elo_t2
            player_history[p].append((match["played_at"], not t1_won))
            level_history[p][level].append(not t1_won)

        if len(t1) == 2:
            pair_history[pkey_t1].append((match["played_at"], t1_won))
        if len(t2) == 2:
            pair_history[pkey_t2].append((match["played_at"], not t1_won))

        # Update H2H
        if t1_won:
            h2h[h2h_key]["t1" if h2h_slot == "t1" else "t2"] += 1
        else:
            h2h[h2h_key]["t2" if h2h_slot == "t1" else "t1"] += 1

    return pd.DataFrame(rows)
