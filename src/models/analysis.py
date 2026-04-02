"""
Analysis functions for dashboard pages beyond prediction.
Covers: ELO timeline, real pairs, H2H records, pair chemistry, tournament history.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.features import ELO_DEFAULT, _expected, _updated_elo

PROCESSED = Path(__file__).parents[2] / "data" / "processed"


def _load_labeled_matches() -> pd.DataFrame:
    return (
        pd.read_parquet(PROCESSED / "matches.parquet")
        .dropna(subset=["winner"])
        .query("winner in ('team_1', 'team_2')")
        .sort_values("played_at")
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def get_elo_timeline() -> pd.DataFrame:
    """ELO rating after each match for every player.

    Returns DataFrame(player_id, name, category, played_at, elo).
    """
    df = _load_labeled_matches()
    df_players = pd.read_parquet(PROCESSED / "players.parquet")

    name_lookup = dict(zip(df_players["player_id"].astype(int), df_players["name"]))
    cat_lookup  = dict(zip(df_players["player_id"].astype(int), df_players["category"]))

    elo  = defaultdict(lambda: ELO_DEFAULT)
    rows = []

    for _, match in df.iterrows():
        t1 = [int(p) for p in [match["t1_p1"], match["t1_p2"]] if pd.notna(p)]
        t2 = [int(p) for p in [match["t2_p1"], match["t2_p2"]] if pd.notna(p)]
        if not t1 or not t2:
            continue

        t1_won  = match["winner"] == "team_1"
        elo_t1  = float(np.mean([elo[p] for p in t1]))
        elo_t2  = float(np.mean([elo[p] for p in t2]))
        exp_t1  = _expected(elo_t1, elo_t2)
        new_t1  = _updated_elo(elo_t1, exp_t1, 1.0 if t1_won else 0.0)
        new_t2  = _updated_elo(elo_t2, 1.0 - exp_t1, 0.0 if t1_won else 1.0)

        for p in t1:
            elo[p] = new_t1
            rows.append({"player_id": p, "name": name_lookup.get(p, str(p)),
                         "category": cat_lookup.get(p, "unknown"),
                         "played_at": match["played_at"], "elo": round(new_t1, 1)})
        for p in t2:
            elo[p] = new_t2
            rows.append({"player_id": p, "name": name_lookup.get(p, str(p)),
                         "category": cat_lookup.get(p, "unknown"),
                         "played_at": match["played_at"], "elo": round(new_t2, 1)})

    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def get_real_pairs(min_matches: int = 3) -> pd.DataFrame:
    """All pairs that have played together at least min_matches times.

    Returns DataFrame(pair_key, names, category, matches_played, wins, win_rate).
    pair_key is a tuple of two sorted player IDs.
    """
    df        = _load_labeled_matches()
    df_players = pd.read_parquet(PROCESSED / "players.parquet")

    name_lookup = dict(zip(df_players["player_id"].astype(int), df_players["name"]))
    cat_lookup  = dict(zip(df_players["player_id"].astype(int), df_players["category"]))

    stats = defaultdict(lambda: {"wins": 0, "total": 0, "category": None})

    for _, match in df.iterrows():
        t1     = [int(p) for p in [match["t1_p1"], match["t1_p2"]] if pd.notna(p)]
        t2     = [int(p) for p in [match["t2_p1"], match["t2_p2"]] if pd.notna(p)]
        t1_won = match["winner"] == "team_1"

        if len(t1) == 2:
            key = tuple(sorted(t1))
            stats[key]["total"]    += 1
            stats[key]["wins"]     += 1 if t1_won else 0
            stats[key]["category"]  = cat_lookup.get(t1[0], "unknown")

        if len(t2) == 2:
            key = tuple(sorted(t2))
            stats[key]["total"]    += 1
            stats[key]["wins"]     += 1 if not t1_won else 0
            stats[key]["category"]  = cat_lookup.get(t2[0], "unknown")

    rows = []
    for key, s in stats.items():
        if s["total"] < min_matches:
            continue
        rows.append({
            "pair_key":       key,
            "names":          " / ".join(name_lookup.get(p, str(p)) for p in key),
            "category":       s["category"],
            "matches_played": s["total"],
            "wins":           s["wins"],
            "win_rate":       round(s["wins"] / s["total"], 3),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("matches_played", ascending=False)
        .reset_index(drop=True)
    )


def get_pairs_faced(pair_ids: tuple[int, int]) -> list[tuple[int, int]]:
    """Return all pair keys that have faced the given pair in a labeled match."""
    df     = _load_labeled_matches()
    target = set(pair_ids)
    faced  = set()

    for _, match in df.iterrows():
        t1 = tuple(sorted(int(p) for p in [match["t1_p1"], match["t1_p2"]] if pd.notna(p)))
        t2 = tuple(sorted(int(p) for p in [match["t2_p1"], match["t2_p2"]] if pd.notna(p)))
        if len(t1) != 2 or len(t2) != 2:
            continue
        if set(t1) == target:
            faced.add(t2)
        elif set(t2) == target:
            faced.add(t1)

    return list(faced)


def get_h2h_record(pair1_ids: tuple[int, int], pair2_ids: tuple[int, int]) -> dict:
    """Full head-to-head record between two specific pairs.

    Returns dict(pair1_names, pair2_names, wins1, wins2, total, matches DataFrame).
    """
    df            = _load_labeled_matches()
    df_players    = pd.read_parquet(PROCESSED / "players.parquet")
    df_tournaments = pd.read_parquet(PROCESSED / "tournaments.parquet")

    name_lookup  = dict(zip(df_players["player_id"].astype(int), df_players["name"]))
    level_lookup = dict(zip(df_tournaments["tournament_id"].astype(int), df_tournaments["level"].fillna("—")))
    tname_lookup = dict(zip(df_tournaments["tournament_id"].astype(int), df_tournaments["name"].fillna("")))

    set1 = set(pair1_ids)
    set2 = set(pair2_ids)

    wins1, wins2 = 0, 0
    match_rows   = []

    for _, match in df.iterrows():
        t1 = {int(p) for p in [match["t1_p1"], match["t1_p2"]] if pd.notna(p)}
        t2 = {int(p) for p in [match["t2_p1"], match["t2_p2"]] if pd.notna(p)}

        if t1 == set1 and t2 == set2:
            p1_won = match["winner"] == "team_1"
        elif t1 == set2 and t2 == set1:
            p1_won = match["winner"] == "team_2"
        else:
            continue

        if p1_won:
            wins1 += 1
        else:
            wins2 += 1

        tid   = match.get("tournament_id")
        level = level_lookup.get(int(tid), "—") if pd.notna(tid) else "—"
        tname = tname_lookup.get(int(tid), "")  if pd.notna(tid) else ""

        winner_ids = pair1_ids if p1_won else pair2_ids
        match_rows.append({
            "date":       match["played_at"],
            "winner":     " / ".join(name_lookup.get(p, str(p)) for p in winner_ids),
            "round":      match.get("round_name", match.get("round")),
            "tournament": tname,
            "level":      level,
        })

    pair1_names = " / ".join(name_lookup.get(p, str(p)) for p in pair1_ids)
    pair2_names = " / ".join(name_lookup.get(p, str(p)) for p in pair2_ids)

    return {
        "pair1_names": pair1_names,
        "pair2_names": pair2_names,
        "wins1":       wins1,
        "wins2":       wins2,
        "total":       wins1 + wins2,
        "matches":     pd.DataFrame(match_rows),
    }


@lru_cache(maxsize=1)
def get_pair_chemistry(min_matches: int = 5) -> pd.DataFrame:
    """Chemistry score for each pair: pair win rate minus average individual win rate.

    Positive = pair performs better together than their solo records suggest.
    Returns DataFrame sorted by chemistry descending.
    """
    df        = _load_labeled_matches()
    df_players = pd.read_parquet(PROCESSED / "players.parquet")

    name_lookup = dict(zip(df_players["player_id"].astype(int), df_players["name"]))
    cat_lookup  = dict(zip(df_players["player_id"].astype(int), df_players["category"]))

    player_wins  = defaultdict(int)
    player_total = defaultdict(int)
    pair_wins    = defaultdict(int)
    pair_total   = defaultdict(int)

    for _, match in df.iterrows():
        t1     = [int(p) for p in [match["t1_p1"], match["t1_p2"]] if pd.notna(p)]
        t2     = [int(p) for p in [match["t2_p1"], match["t2_p2"]] if pd.notna(p)]
        t1_won = match["winner"] == "team_1"

        for p in t1:
            player_total[p] += 1
            player_wins[p]  += 1 if t1_won else 0
        for p in t2:
            player_total[p] += 1
            player_wins[p]  += 1 if not t1_won else 0

        if len(t1) == 2:
            key = tuple(sorted(t1))
            pair_total[key] += 1
            pair_wins[key]  += 1 if t1_won else 0
        if len(t2) == 2:
            key = tuple(sorted(t2))
            pair_total[key] += 1
            pair_wins[key]  += 1 if not t1_won else 0

    rows = []
    for key, total in pair_total.items():
        if total < min_matches:
            continue

        pair_wr   = pair_wins[key] / total
        ind_wrs   = [player_wins[p] / player_total[p] for p in key if player_total[p] > 0]
        avg_ind   = float(np.mean(ind_wrs)) if ind_wrs else 0.5
        chemistry = pair_wr - avg_ind

        rows.append({
            "pair":                    " / ".join(name_lookup.get(p, str(p)) for p in key),
            "category":                cat_lookup.get(key[0], "unknown"),
            "matches_together":        total,
            "pair_win_rate":           round(pair_wr, 3),
            "avg_individual_win_rate": round(avg_ind, 3),
            "chemistry":               round(chemistry, 3),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("chemistry", ascending=False)
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def get_tournament_history() -> pd.DataFrame:
    """All tournaments with match counts, categories, and final winner."""
    df_matches     = pd.read_parquet(PROCESSED / "matches.parquet")
    df_players     = pd.read_parquet(PROCESSED / "players.parquet")
    df_tournaments = pd.read_parquet(PROCESSED / "tournaments.parquet")

    name_lookup = dict(zip(df_players["player_id"].astype(int), df_players["name"]))

    finals = (
        df_matches
        .dropna(subset=["winner"])
        .query("winner in ('team_1', 'team_2') and round == 1")
        .copy()
    )

    def winner_names(row):
        ids = [row["t1_p1"], row["t1_p2"]] if row["winner"] == "team_1" else [row["t2_p1"], row["t2_p2"]]
        return " / ".join(name_lookup.get(int(p), "?") for p in ids if pd.notna(p))

    finals["winner_names"] = finals.apply(winner_names, axis=1)
    winner_lookup = dict(zip(
        zip(finals["tournament_id"].astype(int), finals["category"]),
        finals["winner_names"],
    ))

    match_counts = (
        df_matches
        .groupby(["tournament_id", "category"])
        .size()
        .reset_index(name="matches")
    )

    df_t = df_tournaments.copy()
    df_t["tournament_id"] = df_t["tournament_id"].astype(int)

    rows = []
    for _, row in match_counts.iterrows():
        tid    = int(row["tournament_id"])
        cat    = row["category"]
        t_info = df_t[df_t["tournament_id"] == tid]
        if t_info.empty:
            continue
        t = t_info.iloc[0]
        rows.append({
            "tournament_id": tid,
            "name":          t.get("name", ""),
            "location":      t.get("location", ""),
            "country":       t.get("country", ""),
            "level":         t.get("level", ""),
            "start_date":    t.get("start_date"),
            "end_date":      t.get("end_date"),
            "category":      cat,
            "matches":       row["matches"],
            "winner":        winner_lookup.get((tid, cat), "—"),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("start_date", ascending=False)
        .reset_index(drop=True)
    )
