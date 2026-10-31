"""P1 baselines on the frozen expanding-window time folds.

Usage
-----
    python3 scripts/baseline.py --config configs/base.yaml

Models
------
* ``global_mean``     : constant expm1(mean(log1p(y_train))) per fold.
* ``hour_mean``       : per-hour expm1(mean(log1p(y_train))) with global fallback.
* ``ridge``           : Ridge on log1p(count), no target aggregates.
* ``ridge_target_enc``: Ridge with fold-safe smoothed target aggregates
                        (diagnostic; shows the effect of target encoding).

All predictions are clipped to >= 0.  Metrics are computed on the pooled
out-of-fold predictions, so each validation row contributes exactly once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src.common.config import load_config, path_of  # noqa: E402
from src.common.metrics import mae, r2, rmse, rmsle  # noqa: E402
from src.common.split import load_folds  # noqa: E402
from src.features.build import FeatureBuilder  # noqa: E402


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmsle": rmsle(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def _hour_of_day(dt: pd.Series) -> np.ndarray:
    return pd.to_datetime(dt).dt.hour.to_numpy()


def _ridge_predict(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    alpha: float,
    use_target_aggregates: bool,
) -> np.ndarray:
    builder = FeatureBuilder(use_target_aggregates=use_target_aggregates)
    builder.fit(train_df.iloc[train_idx])
    x_train = builder.transform(train_df.iloc[train_idx])
    x_valid = builder.transform(valid_df.iloc[valid_idx])

    cat_cols = [c for c in x_train.columns if x_train[c].dtype.name in {"category", "object"}]
    num_cols = [c for c in x_train.columns if c not in cat_cols]
    # Convert pandas category dtype to strings so OneHotEncoder sees a stable,
    # plain object dtype on both train and validation frames.
    for c in cat_cols:
        x_train[c] = x_train[c].astype(str)
        x_valid[c] = x_valid[c].astype(str)

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])
    model.fit(x_train, np.log1p(train_df.iloc[train_idx]["count"].to_numpy(dtype="float64")))
    pred = np.expm1(model.predict(x_valid))
    return np.clip(pred.astype("float64"), 0.0, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 baseline evaluation")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = path_of(cfg, "processed_dir")
    reports = path_of(cfg, "reports")
    train_path = processed / "train.parquet"
    if not train_path.exists():
        raise SystemExit(
            "[baseline] processed train not found; run "
            "`python3 -m src.data.make_dataset --config configs/base.yaml` first"
        )

    train = pd.read_parquet(train_path)
    folds, payload = load_folds(cfg)
    if len(folds) != 5:
        raise SystemExit(f"[baseline] expected 5 folds, got {len(folds)}")
    if len(train) != 10_886:
        raise SystemExit(f"[baseline] unexpected train rows: {len(train)}")

    y = train["count"].to_numpy(dtype="float64")
    hour = _hour_of_day(train["datetime"])
    alpha = float(cfg["model"]["ridge"]["alpha"])

    model_names = ["global_mean", "hour_mean", "ridge", "ridge_target_enc"]
    oof = {name: np.full(len(train), np.nan, dtype="float64") for name in model_names}
    fold_rows: list[dict] = []

    print("=" * 78)
    print("P1 baselines — frozen expanding-window time folds")
    print("=" * 78)
    print(f"train rows={len(train):,}  folds={len(folds)}  ridge alpha={alpha}")
    print(f"split version={payload.get('version')}  strategy={payload.get('strategy')}")
    print("target transform: log1p for fitting; predictions back-transformed and clipped at 0")

    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        y_tr = y[train_idx]
        y_va = y[valid_idx]
        log_global = float(np.mean(np.log1p(y_tr)))
        hour_mean_log = (
            pd.Series(np.log1p(y_tr))
            .groupby(hour[train_idx])
            .mean()
        )

        preds = {
            "global_mean": np.full(len(valid_idx), np.expm1(log_global)),
            "hour_mean": np.expm1(
                pd.Series(hour[valid_idx]).map(hour_mean_log).fillna(log_global).to_numpy()
            ),
            "ridge": _ridge_predict(
                train, train, train_idx, valid_idx, alpha, use_target_aggregates=False
            ),
            "ridge_target_enc": _ridge_predict(
                train, train, train_idx, valid_idx, alpha, use_target_aggregates=True
            ),
        }

        for name, pred in preds.items():
            pred = np.clip(np.asarray(pred, dtype="float64"), 0.0, None)
            oof[name][valid_idx] = pred
            m = _metric_row(y_va, pred)
            fold_rows.append({"model": name, "fold": fold_id, "valid_rows": int(len(valid_idx)), **m})

        fold_best = min(
            ((name, _metric_row(y_va, np.clip(preds[name], 0, None))["rmsle"]) for name in model_names),
            key=lambda kv: kv[1],
        )
        print(
            f"[fold {fold_id}] train={len(train_idx):5d} valid={len(valid_idx):4d} "
            f"best={fold_best[0]} rmsle={fold_best[1]:.4f}"
        )

    mask = ~np.isnan(oof["global_mean"])
    if int(mask.sum()) != len(np.unique(np.concatenate([v for _, v in folds]))):
        raise SystemExit("[baseline] OOF coverage does not match the union of validation folds")
    print(f"OOF coverage: {int(mask.sum()):,} rows ({int(mask.sum()) / len(train):.2%} of train)")

    summary_rows = []
    for name in model_names:
        pred = oof[name][mask]
        m = _metric_row(y[mask], pred)
        fm = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        summary_rows.append(
            {
                "model": name,
                "cv_rmsle": m["rmsle"],
                "cv_mae": m["mae"],
                "cv_rmse": m["rmse"],
                "cv_r2": m["r2"],
                "fold_rmsle_mean": float(fm["rmsle"].mean()),
                "fold_rmsle_std": float(fm["rmsle"].std(ddof=0)),
                "oof_rows": int(mask.sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_rows)
    reports.mkdir(parents=True, exist_ok=True)
    summary_path = reports / "p1_baseline.csv"
    folds_path = reports / "p1_baseline_folds.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    folds_df.to_csv(folds_path, index=False, float_format="%.6f")

    oof_df = pd.DataFrame({"row": np.flatnonzero(mask), "y": y[mask]})
    for name in model_names:
        oof_df[name] = oof[name][mask]
    oof_path = reports / "p1_baseline_oof.parquet"
    oof_df.to_parquet(oof_path, index=False)

    print("-" * 78)
    print("Pooled OOF metrics (primary ranking = RMSLE):")
    print(
        summary[["model", "cv_rmsle", "cv_mae", "cv_rmse", "cv_r2", "fold_rmsle_mean", "fold_rmsle_std"]]
        .sort_values("cv_rmsle")
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print("-" * 78)
    print("Per-fold RMSLE:")
    pivot = folds_df.pivot(index="fold", columns="model", values="rmsle")[model_names]
    print(pivot.to_string(float_format=lambda x: f"{x:.6f}"))
    print("-" * 78)
    print(f"[baseline] wrote {summary_path.relative_to(ROOT)}")
    print(f"[baseline] wrote {folds_path.relative_to(ROOT)}")
    print(f"[baseline] wrote {oof_path.relative_to(ROOT)}")
    print("[baseline] done")


if __name__ == "__main__":
    main()
