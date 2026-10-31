# Bike Sharing Demand — Count Regression / Time Series

Kaggle Bike Sharing Demand: predict hourly bike rental `count` from calendar and weather features.
The data is split by time: train = first 19 days of each month, test = 20th–end of each month.

## Data

- `data/raw/train.csv`: 10,886 rows, Kaggle training split
- `data/raw/test.csv`: 6,493 rows, Kaggle test split
- `data/raw/sample_submission.csv`
- `data/raw/hour.csv`: full UCI data (17,379 rows), reference/EDA only

## Metrics

- Primary: RMSLE
- Secondary: MAE, RMSE, R²
- Predictions clipped to >= 0; `casual` and `registered` are never used as features.

## Commands

```bash
make data      # raw -> processed + time folds
make eda       # P1 EDA figures
make train     # P2 single-model time CV
make compare   # P2 model comparison (mean/ridge/rf/lgbm/lgbm_log)
make predict   # P2 final submission + final OOF
make eval      # P3 metrics/plots
make shap      # P3 SHAP
make badcase   # P3 badcase
make report    # P3 report (generated)
make verify    # acceptance checks
make test      # pytest
make all       # full pipeline
```

## Layout

```
src/common/    config, metrics, split loading
src/data/      data pipeline (P1)
src/features/  feature engineering (P1)
src/models/    training/prediction (P2)
src/eval/      evaluation/interpretation (P3)
scripts/       CLI entry points
reports/       reports, figures, exp_log, OOF
splits/        frozen time folds
```
