"""
Functions to flatten nested API response dicts into flat records suitable for DataFrames.
"""
from __future__ import annotations


def _parse_set_value(s: str) -> int:
    """Extract numeric score, stripping tiebreak suffix: '6(5)' → 6, '10' → 10."""
    try:
        return int(str(s).split("(")[0])
    except (ValueError, AttributeError):
        return 0


def parse_score(score_list: list | None) -> dict:
    """Parse a raw score list into set counts and a structured breakdown.

    Example input:  [{'team_1': '6(5)', 'team_2': '7'}, {'team_1': '6', 'team_2': '3'}]
    Example output: {'sets_won_t1': 1, 'sets_won_t2': 1, 'n_sets': 2}
    """
    if not score_list:
        return {"sets_won_t1": None, "sets_won_t2": None, "n_sets": None}

    sets_t1 = 0
    sets_t2 = 0
    for s in score_list:
        if not isinstance(s, dict):
            continue
        t1 = _parse_set_value(s.get("team_1", 0))
        t2 = _parse_set_value(s.get("team_2", 0))
        if t1 > t2:
            sets_t1 += 1
        elif t2 > t1:
            sets_t2 += 1

    return {"sets_won_t1": sets_t1, "sets_won_t2": sets_t2, "n_sets": len(score_list)}


def flatten_match(match: dict) -> dict:
    """Flatten a MatchResource dict into a single-level record.

    Key output columns:
        match_id, tournament_id, category, round, round_name, played_at,
        status, winner, duration,
        t1_p1, t1_p2, t2_p1, t2_p2,
        sets_won_t1, sets_won_t2, n_sets
    """
    players = match.get("players") or {}
    team_1 = players.get("team_1") or []
    team_2 = players.get("team_2") or []

    # Extract tournament_id from connections path e.g. '/api/tournaments/728'
    connections = match.get("connections") or {}
    tournament_path = connections.get("tournament", "")
    try:
        tournament_id = int(tournament_path.rstrip("/").split("/")[-1])
    except (ValueError, IndexError):
        tournament_id = None

    score_info = parse_score(match.get("score"))

    return {
        "match_id": match.get("id"),
        "tournament_id": tournament_id,
        "category": match.get("category"),
        "round": match.get("round"),
        "round_name": match.get("round_name"),
        "played_at": match.get("played_at"),
        "status": match.get("status"),
        "winner": match.get("winner"),       # 'team_1', 'team_2', or None
        "duration": match.get("duration"),
        "t1_p1": team_1[0]["id"] if len(team_1) > 0 else None,
        "t1_p2": team_1[1]["id"] if len(team_1) > 1 else None,
        "t2_p1": team_2[0]["id"] if len(team_2) > 0 else None,
        "t2_p2": team_2[1]["id"] if len(team_2) > 1 else None,
        **score_info,
    }


def flatten_player(player: dict) -> dict:
    """Flatten a PlayerResource dict into a single-level record."""
    return {
        "player_id": player.get("id"),
        "name": player.get("name"),
        "short_name": player.get("short_name"),
        "category": player.get("category"),
        "ranking": player.get("ranking"),
        "points": player.get("points"),
        "nationality": player.get("nationality"),
        "birthdate": player.get("birthdate"),
        "age": player.get("age"),
        "height": player.get("height"),
        "hand": player.get("hand"),
        "side": player.get("side"),
    }


def flatten_tournament(tournament: dict) -> dict:
    """Flatten a TournamentResource dict into a single-level record."""
    connections = tournament.get("connections") or {}
    season_path = connections.get("season", "")
    try:
        season_id = int(season_path.rstrip("/").split("/")[-1])
    except (ValueError, IndexError):
        season_id = None

    return {
        "tournament_id": tournament.get("id"),
        "season_id": season_id,
        "name": tournament.get("name"),
        "location": tournament.get("location"),
        "country": tournament.get("country"),
        "level": tournament.get("level"),
        "status": tournament.get("status"),
        "start_date": tournament.get("start_date"),
        "end_date": tournament.get("end_date"),
    }
