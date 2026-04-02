# Padel Analytics

A data science project analysing professional padel match data — collecting real match results from a REST API, engineering predictive features, training ML models, and serving predictions through an interactive Streamlit dashboard.

Built as a CV project before starting a Data Science for Business master's at BI Norwegian Business School.

## Project Roadmap

| Phase | Status | Description |
|---|---|---|
| 1. API exploration | ✅ Done | Understand data shape and available endpoints |
| 2. Data collection | ✅ Done | Paginate and store all match/player/tournament data as parquet |
| 3. Feature engineering | ✅ Done | ELO ratings, rolling win rates, pair chemistry, H2H, form streak |
| 4. ML models | ✅ Done | Match outcome prediction with LR, Random Forest, XGBoost + SHAP |
| 5. Interactive dashboard | ✅ Done | Streamlit app — ELO rankings and live match predictor |
| 6. Player profiles | ⬜ Planned | Form over time, ELO history curve, recent matches |
| 7. Hyperparameter tuning | ⬜ Planned | Optuna search to squeeze additional performance |
| 8. Deployment | ⬜ Planned | Host on Streamlit Community Cloud for public sharing |

## How Prediction Works

When you select two pairs of players in the Match Predictor and hit **Predict**, this is what happens:

1. **Replay match history** — the app replays every labeled match chronologically to rebuild the current state for each player: ELO rating, win/loss history, pair history, and level-specific records.

2. **Build pre-match features** — for the selected matchup, 33 features are computed using only information that would have been available before the match:
   - **ELO** — each team's average ELO rating and the difference between them
   - **Rolling win rate** — win rate over last 20 matches per player
   - **Form streak** — current consecutive win/loss run
   - **Pair chemistry** — how often this exact pair has won together
   - **Head-to-head** — direct record between these two pairs
   - **Level win rate** — win rate specifically at this tournament tier
   - **Days rest** — days since each player last played
   - **Ranking** — official ranking and difference
   - **Match context** — tournament level weight, round, category

3. **XGBoost prediction** — the 33 features are fed into an XGBoost classifier trained on 891 labeled matches (free API tier). The model outputs a win probability for each team.

> **Data constraint:** The free API tier hides match winners for ~83% of matches. The model is trained on 891 labeled matches with ~75.4% test accuracy, beating an ELO-only baseline of 64.2%. More data from a paid API plan would improve confidence significantly.

## Model Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| ELO baseline | 64.2% | — |
| Logistic Regression | 74.3% | 0.838 |
| Random Forest | 75.4% | 0.831 |
| XGBoost | 75.4% | 0.838 |

All models beat ELO by ~11 percentage points using only free-tier data. SHAP analysis shows `ranking_diff`, `elo_diff`, and `matches_played_diff` as the strongest predictors.

## Running the Dashboard

```bash
streamlit run app/app.py
```

## Project Structure

```
padel-analytics/
├── app/
│   ├── app.py                        # Streamlit entry point (overview + stats)
│   └── pages/
│       ├── 1_ELO_Rankings.py         # ELO leaderboard and Plotly charts
│       └── 2_Match_Predictor.py      # Live win probability predictor
├── notebooks/
│   ├── 01_explore_api.ipynb          # API exploration and data shape
│   ├── 02_collect_data.ipynb         # Full data collection → parquet
│   ├── 03_feature_engineering.ipynb  # Feature computation and ELO baseline
│   └── 04_modelling.ipynb            # Model training, comparison, SHAP
├── src/
│   ├── api/
│   │   └── client.py                 # PadelAPIClient (auth, pagination, rate limiting)
│   ├── processing/
│   │   ├── flatten.py                # Flatten nested API responses to flat records
│   │   └── features.py               # Leak-free feature engineering pipeline
│   └── models/
│       └── predict.py                # Reusable prediction engine (load, rank, predict)
├── data/
│   ├── raw/                          # JSON cache (auto-managed by client)
│   └── processed/                    # Parquet files (matches, players, tournaments, features)
├── .env                              # API key (not committed)
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

Then run the notebooks in order (01 → 02 → 03 → 04), then launch the dashboard.

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

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | required | Bearer token for padelapi.org |
| `PADELAPI_BASE_URL` | `https://padelapi.org` | API base URL |
| `PADELAPI_TIMEOUT_SECONDS` | `30` | Request timeout |
| `PADELAPI_OFFLINE` | `0` | Set to `1` to read from cache only |
| `PADELAPI_USER_AGENT` | `curl/8.0.0` | User-Agent header sent with requests |
