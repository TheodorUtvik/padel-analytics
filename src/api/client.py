import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://padelapi.org"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_USER_AGENT = "curl/8.0.0"

BASE_URL = os.getenv("PADELAPI_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
RAW_DATA_DIR = Path(__file__).parents[2] / "data" / "raw"


class PadelAPIClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: float | None = None):
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY not found in .env")
        self.base_url = (base_url or os.getenv("PADELAPI_BASE_URL", DEFAULT_BASE_URL)).strip().rstrip("/")
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(os.getenv("PADELAPI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
        )
        self.session = requests.Session()
        user_agent = os.getenv("PADELAPI_USER_AGENT", DEFAULT_USER_AGENT)
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": user_agent,
            }
        )

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 403:
                hint = (
                    "403 Forbidden from the API. This often happens when a gateway/WAF blocks the default "
                    "python HTTP client headers. Try setting `PADELAPI_USER_AGENT=curl/8.0.0` (or a browser UA) "
                    "and rerun."
                )
                raise PermissionError(f"{hint} URL={url}") from e
            raise
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Connection error calling {url}. "
                "If you see a NameResolutionError/Failed to resolve host, your environment "
                "can't resolve the Padel API hostname (DNS/network/VPN issue)."
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Timed out calling {url} after {self.timeout_seconds}s") from e

    def _post(self, endpoint: str, body: dict = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, json=body, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError:
            raise
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Connection error calling {url}.") from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Timed out calling {url} after {self.timeout_seconds}s") from e

    def _get_cached(self, cache_file: str, endpoint: str, params: dict = None) -> dict:
        """Fetch from cache if available, otherwise call API and save."""
        path = RAW_DATA_DIR / cache_file
        if path.exists():
            with open(path) as f:
                return json.load(f)
        offline = os.getenv("PADELAPI_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
        if offline:
            raise RuntimeError(
                f"Offline mode enabled (PADELAPI_OFFLINE=1) and cache miss for {path}. "
                "Disable offline mode or add cached data under data/raw/."
            )
        data = self._get(endpoint, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    # --- Seasons ---

    def get_seasons(self) -> dict:
        """List all seasons."""
        return self._get_cached("seasons.json", "/api/seasons")

    def get_season_tournaments(self, season_id: int) -> dict:
        """List all tournaments in a season."""
        return self._get_cached(
            f"seasons/{season_id}/tournaments.json",
            f"/api/seasons/{season_id}/tournaments",
        )

    # --- Players ---

    def get_players(self, name: str = None, nationality: str = None,
                    category: str = None, side: str = None) -> dict:
        """List players. Supports filtering by name, nationality, category (men/women), side (drive/backhand)."""
        params = {}
        if name:
            params["name"] = name
        if nationality:
            params["nationality"] = nationality
        if category:
            params["category"] = category
        if side:
            params["side"] = side
        return self._get("/api/players", params)

    def get_player(self, player_id: int) -> dict:
        """Show a single player's profile."""
        return self._get_cached(f"players/{player_id}.json", f"/api/players/{player_id}")

    def get_player_matches(self, player_id: int, after_date: str = None,
                           before_date: str = None, round: int = None) -> dict:
        """List matches for a specific player."""
        params = {}
        if after_date:
            params["after_date"] = after_date
        if before_date:
            params["before_date"] = before_date
        if round:
            params["round"] = round
        return self._get(f"/api/players/{player_id}/matches", params)

    # --- Matches ---

    def get_matches(self, after_date: str = None, before_date: str = None,
                    round: int = None, category: str = None) -> dict:
        """List matches. Supports filtering by date range, round, and category (men/women).
        Round values: 1=Final, 2=Semi, 4=QF, 8=R16, 16=R32, 32=R64.
        """
        params = {}
        if after_date:
            params["after_date"] = after_date
        if before_date:
            params["before_date"] = before_date
        if round:
            params["round"] = round
        if category:
            params["category"] = category
        return self._get("/api/matches", params)

    def get_match(self, match_id: int) -> dict:
        """Show a single match."""
        return self._get_cached(f"matches/{match_id}.json", f"/api/matches/{match_id}")

    def get_head_to_head(self, team_1: list[int], team_2: list[int] = None) -> dict:
        """Search head-to-head matches between players/pairs.
        team_1: list of 1-2 player IDs.
        team_2: list of 1-2 player IDs (optional — omit to see all matches for team_1).
        """
        body = {"team_1": team_1}
        if team_2:
            body["team_2"] = team_2
        return self._post("/api/matches/headtohead", body)

    # --- Tournaments ---

    def get_tournaments(self, name: str = None, country: str = None, level: str = None,
                        after_date: str = None, before_date: str = None) -> dict:
        """List tournaments. Level options: major, p1, p2, fip_gold, fip_platinum, fip_other,
        wpt_1000, wpt_500, wpt_final, wpt_master, finals.
        """
        params = {}
        if name:
            params["name"] = name
        if country:
            params["country"] = country
        if level:
            params["level"] = level
        if after_date:
            params["after_date"] = after_date
        if before_date:
            params["before_date"] = before_date
        return self._get("/api/tournaments", params)

    def get_tournament(self, tournament_id: int) -> dict:
        """Show a single tournament."""
        return self._get_cached(f"tournaments/{tournament_id}.json", f"/api/tournaments/{tournament_id}")

    def get_tournament_matches(self, tournament_id: int, round: int = None, category: str = None) -> dict:
        """List all matches in a tournament."""
        params = {}
        if round:
            params["round"] = round
        if category:
            params["category"] = category
        return self._get(f"/api/tournaments/{tournament_id}/matches", params)

    # --- Pairs ---

    def get_pair(self, p1_id: int, p2_id: int) -> dict:
        """Show profile for a specific pair."""
        return self._get_cached(f"pairs/{p1_id}-{p2_id}.json", f"/api/pairs/{p1_id}-{p2_id}")

    def get_pair_matches(self, p1_id: int, p2_id: int, after_date: str = None,
                         before_date: str = None, round: int = None) -> dict:
        """List matches for a specific pair."""
        params = {}
        if after_date:
            params["after_date"] = after_date
        if before_date:
            params["before_date"] = before_date
        if round:
            params["round"] = round
        return self._get(f"/api/pairs/{p1_id}-{p2_id}/matches", params)
