# Padel Analytics

A spare-time data science project I built before starting a Data Science for Business master's at BI Norwegian Business School.

The idea: collect real professional padel match data from an API, engineer predictive features, train ML models to predict match outcomes, and serve the results through an interactive dashboard.

## What it does

- Collects match, player, and tournament data from [padelapi.org](https://padelapi.org) (free tier)
- Engineers 33 pre-match features — ELO ratings, rolling win rates, pair chemistry, head-to-head records, form streaks, and more — with no data leakage
- Trains and compares Logistic Regression, Random Forest, and XGBoost models
- Serves predictions through a Streamlit dashboard where you can pick any two pairs of players and get a win probability

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| ELO baseline | 64.2% | — |
| Logistic Regression | 74.3% | 0.838 |
| Random Forest | 75.4% | 0.831 |
| XGBoost | 75.4% | 0.838 |

All models beat the ELO-only baseline by ~11 percentage points.

> The free API tier hides match winners for ~83% of matches, so the model trains on 891 labeled matches out of 5170 total. Results are directionally strong but confidence intervals are wide.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` file with your API key:

```
API_KEY=your_api_key_here
```

Run the notebooks in order (01 → 02 → 03 → 04), then launch the dashboard:

```bash
streamlit run app/app.py
```

## Project structure

```
notebooks/        # Data collection, feature engineering, modelling
src/              # API client, feature pipeline, prediction engine
app/              # Streamlit dashboard
data/processed/   # Parquet files (not committed)
```

For more detail on the technical implementation see [TECHNICAL.md](TECHNICAL.md).
