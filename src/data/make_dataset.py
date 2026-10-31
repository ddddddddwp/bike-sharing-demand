"""Build processed Bike Sharing Demand data and validate the frozen time folds.

Usage
-----
    python3 -m src.data.make_dataset --config configs/base.yaml

The raw Kaggle files are sorted by ``datetime`` and written to parquet.  The
frozen split file ``splits/time_folds.json`` is *read-only*: when it already
exists it is validated, never regenerated.  If it is missing, a fallback
expanding-window split matching PROJECT_SPEC.md is created once.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config, path_of

EXPECTED_TRAIN_ROWS = 10_886
EXPECTED_TEST_ROWS = 6_493
EXPECTED_FOLDS = 5

# (train_start, train_end), (valid_start, valid_end) -- frozen in PROJECT_SPEC.md
EXPECTED_FOLD_MONTHS = [
    (("2011-01", "2011-12"), ("2012-01", "2012-02")),
    (("2011-01", "2012-02"), ("2012-03", "2012-04")),
    (("2011-01", "2012-04"), ("2012-05", "2012-06")),
    (("2011-01", "2012-06"), ("2012-07", "2012-08")),
    (("2011-01", "2012-08"), ("2012-09", "2012-12")),
]

TRAIN_COLUMNS = [
    "datetime", "season", "holiday", "workingday", "weather", "temp",
    "atemp", "humidity", "windspeed", "casual", "registered", "count",
]
TEST_COLUMNS = [
    "datetime", "season", "holiday", "workingday", "weather", "temp",
    "atemp", "humidity", "windspeed",
]


def _fail(msg: str) -> None:
    raise SystemExit(f"[make_dataset] ERROR: {msg}")


def _month_range(start: str, end: str) -> list[str]:
    periods = pd.period_range(start=start, end=end, freq="M")
    return [str(p) for p in periods]


def _read_raw(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        _fail(f"raw file not found: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        _fail(f"{path.name} missing columns: {missing_cols}")
    if df["datetime"].isna().any():
        _fail(f"{path.name} contains unparsable datetime values")
    if df["datetime"].duplicated().any():
        n = int(df["datetime"].duplicated().sum())
        _fail(f"{path.name} contains {n} duplicate datetime rows")
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        _fail(f"{path.name} contains {n_missing} missing values")
    return df.sort_values("datetime", kind="mergesort").reset_index(drop=True)


def _validate_processed(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if len(train) != EXPECTED_TRAIN_ROWS:
        _fail(f"train rows {len(train)} != {EXPECTED_TRAIN_ROWS}")
    if len(test) != EXPECTED_TEST_ROWS:
        _fail(f"test rows {len(test)} != {EXPECTED_TEST_ROWS}")
    if not train["datetime"].is_monotonic_increasing:
        _fail("processed train is not sorted by datetime")
    if not test["datetime"].is_monotonic_increasing:
        _fail("processed test is not sorted by datetime")
    forbidden = {"casual", "registered", "count"}
    leaked = forbidden & set(test.columns)
    if leaked:
        _fail(f"test frame unexpectedly contains target/leaky columns: {sorted(leaked)}")
    if "count" not in train.columns:
        _fail("train frame is missing the target column `count`")


def _make_folds(train: pd.DataFrame) -> dict:
    """Fallback split generation; only used when the frozen file is absent."""
    months = train["datetime"].dt.to_period("M").astype(str)
    folds = []
    for i, ((tr_start, tr_end), (va_start, va_end)) in enumerate(EXPECTED_FOLD_MONTHS):
        train_months = _month_range(tr_start, tr_end)
        valid_months = _month_range(va_start, va_end)
        train_idx = np.flatnonzero(months.isin(train_months).to_numpy())
        valid_idx = np.flatnonzero(months.isin(valid_months).to_numpy())
        folds.append(
            {
                "fold": i,
                "train_idx": train_idx.astype(int).tolist(),
                "valid_idx": valid_idx.astype(int).tolist(),
                "train_months": train_months,
                "valid_months": valid_months,
            }
        )
    return {
        "version": "time_v1",
        "n_rows": int(len(train)),
        "n_splits": EXPECTED_FOLDS,
        "strategy": "expanding_month",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folds": folds,
    }


def _validate_folds(payload: dict, train: pd.DataFrame) -> None:
    if payload.get("n_splits") != EXPECTED_FOLDS:
        _fail(f"split n_splits={payload.get('n_splits')} != {EXPECTED_FOLDS}")
    if payload.get("n_rows") not in (None, len(train)):
        _fail(f"split n_rows={payload.get('n_rows')} != processed rows {len(train)}")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != EXPECTED_FOLDS:
        _fail(f"split must contain exactly {EXPECTED_FOLDS} folds")

    dt = train["datetime"].reset_index(drop=True)
    months = dt.dt.to_period("M").astype(str)
    n = len(train)
    previous_train: set[int] | None = None

    for i, fold in enumerate(folds):
        tr_idx = np.asarray(fold.get("train_idx", []), dtype=int)
        va_idx = np.asarray(fold.get("valid_idx", []), dtype=int)
        if fold.get("fold") != i:
            _fail(f"fold {i} has fold id {fold.get('fold')}")
        if len(tr_idx) == 0 or len(va_idx) == 0:
            _fail(f"fold {i} has an empty train or valid index list")
        if tr_idx.min() < 0 or va_idx.min() < 0 or tr_idx.max() >= n or va_idx.max() >= n:
            _fail(f"fold {i} index outside [0, {n})")
        if len(np.unique(tr_idx)) != len(tr_idx) or len(np.unique(va_idx)) != len(va_idx):
            _fail(f"fold {i} contains duplicate indices")
        overlap = np.intersect1d(tr_idx, va_idx)
        if overlap.size:
            _fail(f"fold {i} train/valid overlap: {overlap.size} rows")

        tr_dt, va_dt = dt.iloc[tr_idx], dt.iloc[va_idx]
        if tr_dt.max() >= va_dt.min():
            _fail(f"fold {i} is not time ordered (train max >= valid min)")
        if tr_idx.max() >= va_idx.min():
            _fail(f"fold {i} positional order violates time order")

        expected_tr = _month_range(*EXPECTED_FOLD_MONTHS[i][0])
        expected_va = _month_range(*EXPECTED_FOLD_MONTHS[i][1])
        actual_tr = sorted(months.iloc[tr_idx].unique().tolist())
        actual_va = sorted(months.iloc[va_idx].unique().tolist())
        if actual_tr != expected_tr:
            _fail(f"fold {i} train months {actual_tr} != expected {expected_tr}")
        if actual_va != expected_va:
            _fail(f"fold {i} valid months {actual_va} != expected {expected_va}")

        train_set = set(tr_idx.tolist())
        if previous_train is not None:
            if not previous_train.issubset(train_set):
                _fail(f"fold {i} train window is not expanding")
            if len(train_set) <= len(previous_train):
                _fail(f"fold {i} train window did not grow")
        previous_train = train_set

        print(
            f"[make_dataset] fold {i}: train={len(tr_idx):5d} rows "
            f"({tr_dt.min():%Y-%m-%d} .. {tr_dt.max():%Y-%m-%d}), "
            f"valid={len(va_idx):4d} rows "
            f"({va_dt.min():%Y-%m-%d} .. {va_dt.max():%Y-%m-%d})"
        )

    covered = np.unique(np.concatenate([np.asarray(f["valid_idx"], dtype=int) for f in folds]))
    if len(covered) != sum(len(f["valid_idx"]) for f in folds):
        _fail("valid folds overlap each other")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed parquet + validate time folds")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = path_of(cfg, "raw_dir")
    processed_dir = path_of(cfg, "processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    train = _read_raw(raw_dir / cfg["data"]["train_file"], TRAIN_COLUMNS)
    test = _read_raw(raw_dir / cfg["data"]["test_file"], TEST_COLUMNS)
    _validate_processed(train, test)

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    print(f"[make_dataset] wrote {train_path.relative_to(ROOT)}  shape={train.shape}")
    print(f"[make_dataset] wrote {test_path.relative_to(ROOT)}  shape={test.shape}")

    split_path = path_of(cfg, "splits")
    if split_path.exists():
        print(f"[make_dataset] frozen split found, validating only: {split_path.relative_to(ROOT)}")
        payload = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _make_folds(train)
        split_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[make_dataset] created fallback split: {split_path.relative_to(ROOT)}")

    _validate_folds(payload, train)
    print(
        f"[make_dataset] OK: train={len(train):,}, test={len(test):,}, "
        f"folds={len(payload['folds'])}, no overlap, time ordered, expanding"
    )


if __name__ == "__main__":
    main()
