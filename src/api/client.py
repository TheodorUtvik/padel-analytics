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
            message = (
                f"Connection error calling {url}. "
                "If you see a NameResolutionError/Failed to resolve host, your environment can't resolve "
                "the Padel API hostname (DNS/network/VPN issue)."
            )
            raise ConnectionError(message) from e
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

    # --- Endpoints ---

    def get_seasons(self) -> dict:
        return self._get_cached("seasons.json", "/api/seasons")

    def get_players(self) -> dict:
        return self._get_cached("players.json", "/api/players")

    def get_player_stats(self, player_id: int, after_date: str = None, before_date: str = None, round: str = None) -> dict:
        params = {}
        if after_date:
            params["after_date"] = after_date
        if before_date:
            params["before_date"] = before_date
        if round:
            params["round"] = round
        cache_key = f"after={after_date}_before={before_date}_round={round}"
        return self._get_cached(
            f"players/{player_id}/stats_{cache_key}.json",
            f"/api/players/{player_id}/stats",
            params,
        )

    def get_matches(self, params: dict = None) -> dict:
        return self._get("/api/matches", params)

    def get_match_stats(self, match_id: int) -> dict:
        return self._get_cached(f"matches/{match_id}/stats.json", f"/api/matches/{match_id}/stats")

    def get_pair_stats(self, p1_id: int, p2_id: int) -> dict:
        return self._get_cached(
            f"pairs/{p1_id}-{p2_id}/stats.json",
            f"/api/pairs/{p1_id}-{p2_id}/stats",
        )
