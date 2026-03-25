# Padel Analytics

A data science side project analysing professional padel match data. Built to explore data science concepts and produce something meaningful for a CV before starting a Data Science for Business master's at BI Norwegian Business School.

## Project Roadmap

| Phase | Status | Description |
|---|---|---|
| 1. API exploration | ✅ Done | Understand data shape and available endpoints |
| 2. Data collection | 🔄 In progress | Paginate and store all match/player data as parquet |
| 3. Feature engineering | ⬜ Planned | Rolling win rates, pair chemistry, ELO ratings |
| 4. ML models | ⬜ Planned | Match outcome prediction, player form classification |
| 5. Interactive dashboard | ⬜ Planned | Streamlit app to explore players, pairs, and predictions |

## Data Source

- **API:** [padelapi.org](https://padelapi.org) — REST API with Bearer token auth
- **Coverage:** Premier Padel, World Padel Tour (2023+), ~900 players, match results
- **Free tier:** 50k requests/month, last 6 months of match data

### Available endpoints (free tier)

| Endpoint | Description |
|---|---|
| `GET /api/seasons` | All seasons |
| `GET /api/players` | Player list (paginated) |
| `GET /api/players/{id}/matches` | Match history for a player |
| `GET /api/matches` | Match results (paginated, filterable) |
| `GET /api/tournaments` | Tournament list |
| `GET /api/tournaments/{id}/matches` | Matches in a tournament |
| `GET /api/pairs/{p1}-{p2}/matches` | Match history for a pair |
| `POST /api/matches/headtohead` | Head-to-head search between players/pairs |

## Project Structure

```
padel-analytics/
├── notebooks/
│   ├── 01_explore_api.ipynb       # API exploration and data shape
│   ├── 02_collect_data.ipynb      # Full data collection → parquet
│   ├── 03_feature_engineering.ipynb  # ELO, rolling stats, pair features (planned)
│   └── 04_modelling.ipynb         # Match outcome prediction (planned)
├── src/
│   ├── api/
│   │   └── client.py              # PadelAPIClient (auth, pagination, rate limiting, caching)
│   ├── processing/
│   │   └── flatten.py             # Flatten nested API responses to flat records
│   └── app/                       # Streamlit dashboard (planned)
├── data/
│   ├── raw/                       # JSON cache (auto-managed by client)
│   └── processed/                 # Parquet files (output of 02_collect_data)
├── .env                           # API key (not committed)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
API_KEY=your_api_key_here
```

Then run the notebooks in order.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | required | Bearer token for padelapi.org |
| `PADELAPI_BASE_URL` | `https://padelapi.org` | API base URL |
| `PADELAPI_TIMEOUT_SECONDS` | `30` | Request timeout |
| `PADELAPI_OFFLINE` | `0` | Set to `1` to read from cache only |
| `PADELAPI_USER_AGENT` | `curl/8.0.0` | User-Agent header sent with requests |
