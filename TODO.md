# TODO

## Dashboard
- [ ] Map tournament level codes (`p1`, `p2`, `fip_other`, `wpt_master`, etc.) to readable names across all pages

## Phase 6 — Player Profiles
- [x] Player profile page with ELO history, win rate by level, rolling form, recent matches

## Phase 7 — Hyperparameter Tuning
- [ ] Use Optuna to tune XGBoost hyperparameters
- [ ] Compare tuned model against current baseline (75.4%)

## Phase 8 — Deployment
- [ ] Deploy dashboard to Streamlit Community Cloud
- [ ] Add public link to README

## Data
- [ ] Await API support response — confirm if €19 Analyst plan unlocks match winners
- [ ] If yes: re-run data collection → feature engineering → retrain models on ~5170 labeled matches
