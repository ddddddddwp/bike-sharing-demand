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


## Validation

- `splits/time_folds.json` (`time_v2`): 5 expanding-window folds by month.
- Validation starts at 2011-07, so OOF covers **8,196 / 10,886 rows (75.3%)**.
- Final model trains on all 10,886 rows and predicts the 6,493-row test set.
- No random KFold; no shuffling across time.

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
make ext       # horizontal extensions: hyperparam / ablation / efficiency / drift / stability
make verify    # acceptance checks
make test      # pytest
make all       # full pipeline
```


## Official Kaggle data

The current `data/raw/train.csv` / `test.csv` were rebuilt from the UCI source using the exact
Kaggle split (10,886 / 6,493 rows). To switch to the official Kaggle files:

1. Kaggle -> Account -> Settings -> API -> Create New API Token
2. Put `kaggle.json` at `~/.kaggle/kaggle.json`
3. Run:

```bash
python3 scripts/fetch_kaggle.py --replace
make data && make compare && make predict && make eval && make report
```


## Horizontal extensions

See `reports/extension_report.md` for:
- hyperparameter sensitivity + convergence
- feature/preprocessing ablation
- model efficiency / lightweight model
- temporal drift (static vs expanding vs sliding)
- multi-seed stability + bootstrap CI


## Reproducibility

See `docs/reproduce.md` for the full reproduction record on the local RTX 5060 machine
(submission is byte-identical, OOF max abs diff = 0, verify 52/52).

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
