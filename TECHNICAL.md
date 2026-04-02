# Technical Reference

## How prediction works

When you select two pairs of players in the Match Predictor and hit **Predict**:

1. **Replay match history** — the app replays every labeled match chronologically to rebuild current state for each player: ELO rating, win/loss history, pair history, and level-specific records.

2. **Build pre-match features** — 33 features are computed using only information available before the match:
   - ELO ratings and difference
   - Rolling win rate (last 20 matches)
   - Form streak (consecutive W/L run)
   - Pair chemistry (win rate for this exact pair together)
   - Head-to-head record between the two pairs
   - Level win rate (win rate at this tournament tier)
   - Days since last match
   - Official ranking and difference
   - Match context (tournament level, round, category)

3. **XGBoost prediction** — the 33 features are passed to an XGBoost classifier trained on all 891 labeled matches, returning a win probability for each team.

## Project structure

```
padel-analytics/
├── app/
│   ├── app.py                        # Streamlit entry point
│   └── pages/
│       ├── 1_ELO_Rankings.py         # ELO leaderboard and charts
│       └── 2_Match_Predictor.py      # Live win probability predictor
├── notebooks/
│   ├── 01_explore_api.ipynb          # API exploration
│   ├── 02_collect_data.ipynb         # Data collection → parquet
│   ├── 03_feature_engineering.ipynb  # Feature computation and ELO baseline
│   └── 04_modelling.ipynb            # Model training, comparison, SHAP
├── src/
│   ├── api/client.py                 # PadelAPIClient (auth, pagination, rate limiting)
│   ├── processing/flatten.py         # Flatten nested API responses
│   ├── processing/features.py        # Leak-free feature engineering pipeline
│   └── models/predict.py             # Reusable prediction engine
├── data/
│   ├── raw/                          # JSON cache (auto-managed)
│   └── processed/                    # Parquet files
└── .env                              # API key (not committed)
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | required | Bearer token for padelapi.org |
| `PADELAPI_BASE_URL` | `https://padelapi.org` | API base URL |
| `PADELAPI_TIMEOUT_SECONDS` | `30` | Request timeout |
| `PADELAPI_OFFLINE` | `0` | Set to `1` to read from cache only |
| `PADELAPI_USER_AGENT` | `curl/8.0.0` | User-Agent header |

## API endpoints used (free tier)

| Endpoint | Description |
|---|---|
| `GET /api/players` | Player list (paginated) |
| `GET /api/matches` | Match results (paginated, filterable) |
| `GET /api/tournaments` | Tournament list |
| `GET /api/tournaments/{id}/matches` | Matches in a tournament |
| `POST /api/matches/headtohead` | Head-to-head between players/pairs |

## Planned

- Player profile page — form over time, ELO history curve, recent matches
- Hyperparameter tuning with Optuna
- Deploy to Streamlit Community Cloud
