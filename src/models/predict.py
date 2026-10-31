"""Fit the final model on all training rows and write the Kaggle submission (P2).

Usage:
    python3 -m src.models.predict --config configs/base.yaml --model lgbm --seed 42
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import load_config, path_of
from src.common.split import load_folds
from src.features.build import FeatureBuilder
from src.models.train import (
    MODEL_NAMES,
    _metrics,
    load_train,
    make_model,
    model_params,
    run_cv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test(cfg: dict) -> pd.DataFrame:
    path = path_of(cfg, "raw_dir") / cfg["data"]["test_file"]
    return pd.read_csv(path)


def _fold_metrics_from_oof(
    oof: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    payload: dict,
) -> list[dict[str, Any]]:
    pred = np.full(len(y), np.nan, dtype="float64")
    pred[oof["row"].to_numpy(dtype=int)] = oof["pred"].to_numpy(dtype="float64")
    out = []
    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        if np.isnan(pred[va_idx]).any():
            raise RuntimeError(f"OOF missing predictions for validation fold {fold_id}")
        fm = _metrics(y[va_idx], pred[va_idx])
        fm.update(
            {
                "fold": fold_id,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "valid_months": payload["folds"][fold_id].get("valid_months", []),
            }
        )
        out.append(fm)
    return out


def _resolve_cv(cfg: dict, model: str, seed: int, log: bool = True) -> dict[str, Any]:
    """Reuse the best comparison run's OOF when possible, otherwise run CV."""
    train = load_train(cfg)
    y = train[cfg["data"]["target"]].to_numpy(dtype="float64")
    folds, payload = load_folds(cfg)
    comp_path = path_of(cfg, "reports") / "p2_model_comparison.csv"

    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        if not comp.empty:
            best = comp.sort_values("cv_rmsle").iloc[0]
            if str(best["model"]) == model:
                oof_path = path_of(cfg, "oof_dir") / f"{best['run_id']}.parquet"
                if oof_path.exists():
                    oof = pd.read_parquet(oof_path)
                    fold_metrics = _fold_metrics_from_oof(oof, y, folds, payload)
                    if log:
                        print(f"[reuse] time-CV OOF for best model {model}: {oof_path}")
                    return {
                        "run_id": str(best["run_id"]),
                        "model": model,
                        "seed": int(seed),
                        "params_hash": str(best["params_hash"]),
                        "n_features": int(best["n_features"]),
                        "n_oof": int(best["n_oof"]),
                        "oof_coverage": float(best["oof_coverage"]),
                        "cv_rmsle": float(best["cv_rmsle"]),
                        "cv_mae": float(best["cv_mae"]),
                        "cv_rmse": float(best["cv_rmse"]),
                        "cv_r2": float(best["cv_r2"]),
                        "fold_metrics": fold_metrics,
                        "oof": oof[["row", "y", "pred"]].copy(),
                        "oof_path": str(oof_path),
                    }
            if log:
                print(
                    f"[warn] requested model={model!r}, but comparison best is "
                    f"{best['model']!r} (RMSLE={float(best['cv_rmsle']):.5f}); "
                    "running a fresh CV for the requested model."
                )

    return run_cv(cfg, model=model, seed=seed, log=log)


def _write_feature_importance(est, cfg: dict, model: str) -> pd.DataFrame | None:
    fi = est.feature_importance()
    if fi is None or fi.empty:
        return None
    out = path_of(cfg, "reports") / "p2_feature_importance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fi.to_csv(out, index=False)
    _plot_feature_importance(fi, path_of(cfg, "figures") / "p2_feature_importance.png", model)
    return fi


def _plot_feature_importance(fi: pd.DataFrame, path: Path, model: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = fi.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(top["feature"], top["importance"], color="#2a7f9e")
    ax.set_xlabel("Importance")
    ax.set_title(f"Final {model} model - top 20 feature importances")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _submission_frame(cfg: dict, test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Build a submission aligned to sample_submission row order when available."""
    test_dt = pd.to_datetime(test["datetime"])
    values = pd.Series(np.asarray(pred, dtype="float64"), index=test_dt)
    sample_path = path_of(cfg, "raw_dir") / cfg["data"]["sample_sub_file"]
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        sample_dt = pd.to_datetime(sample["datetime"])
        missing = ~sample_dt.isin(values.index)
        if missing.any():
            raise RuntimeError(f"{int(missing.sum())} sample_submission datetimes missing from test predictions")
        sub = pd.DataFrame(
            {
                "datetime": sample["datetime"].astype(str),
                "count": values.reindex(sample_dt).to_numpy(dtype="float64"),
            }
        )
    else:
        sub = pd.DataFrame(
            {
                "datetime": test_dt.dt.strftime("%Y-%m-%d %H:%M:%S"),
                "count": np.asarray(pred, dtype="float64"),
            }
        )
    sub["count"] = np.clip(sub["count"], 0.0, None)
    return sub


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def predict_test(cfg: dict, model: str = "lgbm", seed: int = 42, log: bool = True) -> dict[str, Any]:
    if model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {model!r}")

    cv = _resolve_cv(cfg, model=model, seed=seed, log=log)

    train = load_train(cfg)
    test = load_test(cfg)
    target = cfg["data"]["target"]
    y = train[target].to_numpy(dtype="float64")

    if log:
        print(f"[final] fitting {model} on all {len(train)} training rows with FeatureBuilder.fit(all)")
    builder = FeatureBuilder().fit(train)
    x_train = builder.transform(train)
    x_test = builder.transform(test)
    est = make_model(cfg, model, seed)
    est.fit(x_train, y)
    pred = np.clip(np.asarray(est.predict(x_test), dtype="float64"), 0.0, None)
    if len(pred) != len(test):
        raise RuntimeError(f"prediction length {len(pred)} != test length {len(test)}")
    if np.isnan(pred).any():
        raise RuntimeError(f"predictions contain {int(np.isnan(pred).sum())} NaNs")

    # Submission
    sub = _submission_frame(cfg, test, pred)
    sub_path = path_of(cfg, "reports") / "submission.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)

    # Final OOF comes from the selected model's time-CV run.
    oof = cv["oof"][["row", "y", "pred"]].copy()
    oof["pred"] = np.clip(oof["pred"].astype("float64"), 0.0, None)
    oof_path = path_of(cfg, "oof_dir") / "final_oof.parquet"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_path, index=False)

    fi = _write_feature_importance(est, cfg, model)

    by_month = (
        pd.DataFrame(
            {
                "month": pd.to_datetime(sub["datetime"]).dt.strftime("%Y-%m"),
                "count": sub["count"].to_numpy(dtype="float64"),
            }
        )
        .groupby("month")["count"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )
    by_month_path = path_of(cfg, "reports") / "p2_submission_by_month.csv"
    by_month.to_csv(by_month_path, index=False)

    stats = {
        "path": str(sub_path),
        "rows": int(len(sub)),
        "min": float(sub["count"].min()),
        "max": float(sub["count"].max()),
        "mean": float(sub["count"].mean()),
        "median": float(sub["count"].median()),
        "std": float(sub["count"].std()),
        "nan": int(sub["count"].isna().sum()),
        "negative": int((sub["count"] < 0).sum()),
        "first_datetime": str(sub["datetime"].iloc[0]),
        "last_datetime": str(sub["datetime"].iloc[-1]),
        "by_month_path": str(by_month_path),
    }
    oof_dt = pd.to_datetime(train.loc[oof["row"], "datetime"])
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "seed": int(seed),
        "run_id": cv["run_id"],
        "params_hash": cv["params_hash"],
        "cv": {
            "rmsle": float(cv["cv_rmsle"]),
            "mae": float(cv["cv_mae"]),
            "rmse": float(cv["cv_rmse"]),
            "r2": float(cv["cv_r2"]),
            "n_oof": int(cv.get("n_oof", len(oof))),
            "oof_coverage": float(cv.get("oof_coverage", len(oof) / len(train))),
        },
        "fold_metrics": cv["fold_metrics"],
        "final_model": {
            "train_rows": int(len(train)),
            "n_features": int(len(builder.feature_names)),
            "params": model_params(cfg, model),
            "feature_builder": {"use_target_aggregates": True, "smooth": 20.0},
        },
        "submission": stats,
        "oof": {
            "path": str(oof_path),
            "rows": int(len(oof)),
            "nan": int(oof["pred"].isna().sum()),
            "row_min": int(oof["row"].min()),
            "row_max": int(oof["row"].max()),
            "first_datetime": str(oof_dt.min()),
            "last_datetime": str(oof_dt.max()),
        },
        "feature_importance": {
            "path": str(path_of(cfg, "reports") / "p2_feature_importance.csv"),
            "top10": [] if fi is None else fi.head(10).to_dict(orient="records"),
        },
    }
    metrics_path = path_of(cfg, "reports") / "p2_final_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if log:
        print(
            f"[submission] {sub_path} | rows={stats['rows']} min={stats['min']:.3f} "
            f"max={stats['max']:.3f} mean={stats['mean']:.3f} nan={stats['nan']}"
        )
        print(f"[final oof] {oof_path} | rows={len(oof)}")
        print(f"[metrics] {metrics_path} | CV RMSLE={cv['cv_rmsle']:.5f} MAE={cv['cv_mae']:.3f} RMSE={cv['cv_rmse']:.3f}")

    return {
        "model": model,
        "seed": int(seed),
        "run_id": cv["run_id"],
        "cv_rmsle": float(cv["cv_rmsle"]),
        "cv_mae": float(cv["cv_mae"]),
        "cv_rmse": float(cv["cv_rmse"]),
        "cv_r2": float(cv["cv_r2"]),
        "submission": stats,
        "submission_path": str(sub_path),
        "oof_path": str(oof_path),
        "metrics_path": str(metrics_path),
        "feature_importance": None if fi is None else fi.head(20).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bike Sharing Demand final training + submission (P2)")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="lgbm", choices=list(MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    predict_test(cfg, model=args.model, seed=args.seed)


if __name__ == "__main__":
    main()
