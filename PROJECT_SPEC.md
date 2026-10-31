# Bike Sharing Demand — Project Spec (shared contract)

> All three roles (P1/P2/P3) work in this repo. Do not change this file without the orchestrator's approval.
> Old airline project is abandoned; this is a fresh project.

## 1. Task

- Dataset: Kaggle **Bike Sharing Demand** (same data as UCI Bike Sharing).
- Task: **count regression + time series** — predict hourly bike rentals `count`.
- Primary metric: **RMSLE** = sqrt(mean((log1p(pred) - log1p(actual))^2)).
- Secondary: MAE, RMSE, R².
- Predictions must be clipped to `>= 0`.
- **Never use `casual` or `registered` as features** (they sum to `count`, target leakage).

## 2. Data

Raw files in `data/raw/`:

| file | rows | columns |
|---|---:|---|
| `train.csv` | 10,886 | `datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, casual, registered, count` |
| `test.csv` | 6,493 | same minus `casual, registered, count` |
| `sample_submission.csv` | 6,493 | `datetime, count` |

Split rule (matches Kaggle): train = first 19 days of each month; test = 20th–end of each month.
Full UCI data (17,379 rows) is available at `data/raw/hour.csv` for EDA/reference only; do not train on it for the competition split.

## 3. Time-based validation

`splits/time_folds.json` contains 5 expanding-window folds by month (rows are positional indices into `data/processed/train.parquet`):

- fold 0: train 2011-01..2011-12, valid 2012-01..2012-02
- fold 1: train 2011-01..2012-02, valid 2012-03..2012-04
- fold 2: train 2011-01..2012-04, valid 2012-05..2012-06
- fold 3: train 2011-01..2012-06, valid 2012-07..2012-08
- fold 4: train 2011-01..2012-08, valid 2012-09..2012-12

Final model: train on all 10,886 rows, predict `test.csv`.
No random KFold. No shuffling across time.

## 4. Shared interfaces (frozen)

### 4.1 `src/common/config.py`

```python
ROOT: Path
def load_config(path="configs/base.yaml") -> dict
def path_of(cfg, key) -> Path
```

### 4.2 `src/common/metrics.py`

```python
def rmsle(y_true, y_pred) -> float
def mae(y_true, y_pred) -> float
def rmse(y_true, y_pred) -> float
def r2(y_true, y_pred) -> float
```

### 4.3 `src/common/split.py`

```python
def load_folds(cfg) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]
```

### 4.4 `src/features/build.py` (owned by P1)

```python
class FeatureBuilder:
    def __init__(self, use_target_aggregates: bool = True)
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder"
    def transform(self, df: pd.DataFrame) -> pd.DataFrame
    @property
    def feature_names(self) -> list[str]
```

- `fit` must only see training rows.
- `transform` returns features only: no `datetime`, `count`, `casual`, `registered`.
- Categorical columns may be `category` dtype; numeric columns numeric.
- Target aggregates (hour mean, weekday mean, etc.) must be computed in `fit` from the training frame only.

### 4.5 `src/models/train.py` (owned by P2)

```python
def run_cv(cfg, model="lgbm", seed=42, log=True) -> dict
# returns {run_id, model, cv_rmsle, cv_mae, cv_rmse, fold_metrics, oof}
# writes reports/oof/{run_id}.parquet with columns: row, y, pred
```

### 4.6 `src/models/predict.py` (owned by P2)

```python
def predict_test(cfg, model="lgbm", seed=42, log=True) -> dict
# writes reports/submission.csv (datetime,count) and reports/oof/final_oof.parquet
```

### 4.7 `src/eval/*` (owned by P3)

- `evaluate.py`: read `reports/oof/final_oof.parquet` + processed train, write `reports/eval/metrics.json`, `summary.md`, residual/actual-vs-pred plots, segment metrics CSV.
- `interpret.py`: SHAP/feature importance for the final LightGBM, write `reports/eval/shap_importance.csv` + plots.
- `badcase.py`: top-50 absolute/log errors, segment error table, write `reports/eval/badcase_top50.csv` + `badcase_summary.md`.

## 5. File ownership (avoid conflicts)

| Owner | Files |
|---|---|
| **P1** | `src/data/**`, `src/features/**`, `scripts/eda.py`, `scripts/build_features.py`, `scripts/baseline.py`, `docs/**`, `reports/p1_*`, `reports/figures/p1_*` |
| **P2** | `src/models/**`, `scripts/train.py`, `scripts/predict.py`, `reports/exp_log.csv`, `reports/p2_*`, `reports/figures/p2_*` |
| **P3** | `src/eval/**`, `scripts/evaluate.py`, `scripts/interpret.py`, `scripts/badcase.py`, `reports/p3_*`, `reports/figures/p3_*` |
| **Shared (do not edit)** | `src/common/**`, `configs/base.yaml`, `splits/**`, `Makefile`, `README.md`, `PROJECT_SPEC.md` |

## 6. Commands

```bash
make data      # raw -> processed parquet + time folds
make eda       # P1 figures
make train     # P2 CV baseline
make predict   # P2 final submission
make eval      # P3 metrics/plots
make shap      # P3 SHAP
make badcase   # P3 badcase
make all
```
