"""Time-based cross-validation training for Bike Sharing Demand (P2).

Usage:
    python3 -m src.models.train --config configs/base.yaml --model lgbm --seed 42
    python3 -m src.models.train --config configs/base.yaml --compare

The feature builder is frozen (owned by P1).  Every fold fits a fresh
``FeatureBuilder`` on the training rows only, so target aggregates never see
validation targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.common.config import load_config, path_of
from src.common.metrics import mae, r2, rmse, rmsle
from src.common.split import load_folds
from src.features.build import FeatureBuilder

MODEL_NAMES = ("mean", "ridge", "rf", "lgbm", "lgbm_log")
COMPARE_MODELS = ("mean", "ridge", "rf", "lgbm", "lgbm_log")


# ---------------------------------------------------------------------------
# Config / hashing helpers
# ---------------------------------------------------------------------------
def model_params(cfg: dict, model: str) -> dict[str, Any]:
    """Return the effective model parameters recorded in the experiment log."""
    if model == "mean":
        return {"strategy": "global_train_mean"}
    if model == "ridge":
        return dict(cfg.get("model", {}).get("ridge", {"alpha": 1.0}))
    if model == "rf":
        return dict(cfg.get("model", {}).get("rf", {}))
    if model in ("lgbm", "lgbm_log"):
        params = dict(cfg.get("model", {}).get("lgbm", {}))
        if model == "lgbm_log":
            params["objective"] = "regression"
            params["target_transform"] = "log1p"
        else:
            params["objective"] = params.get("objective", "poisson")
            params["target_transform"] = "none"
        return params
    raise ValueError(f"unknown model: {model}")


def params_hash(cfg: dict, model: str) -> str:
    payload = {
        "model": model,
        "params": model_params(cfg, model),
        "features": {"builder": "FeatureBuilder", "use_target_aggregates": True, "smooth": 20.0},
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _make_run_id(model: str, seed: int, phash: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model}_s{seed}_{stamp}_{phash}"


# ---------------------------------------------------------------------------
# Fold-safe feature encoding
# ---------------------------------------------------------------------------
def _cat_cols(x: pd.DataFrame) -> list[str]:
    return [
        c
        for c in x.columns
        if isinstance(x[c].dtype, pd.CategoricalDtype) or x[c].dtype == object
    ]


def _num_cols(x: pd.DataFrame, cats: list[str]) -> list[str]:
    return [c for c in x.columns if c not in cats]


def _fit_levels(x: pd.DataFrame, cats: list[str]) -> dict[str, list[str]]:
    return {c: sorted(pd.unique(x[c].astype(str)).tolist()) for c in cats}


def _apply_levels(x: pd.DataFrame, cats: list[str], levels: dict[str, list[str]]) -> pd.DataFrame:
    """Cast categoricals to the exact levels seen during fit (unseen -> NaN)."""
    out = x.copy()
    for c in cats:
        out[c] = pd.Categorical(out[c].astype(str), categories=levels[c])
    return out


def _matrix_rf(x: pd.DataFrame, cats: list[str], nums: list[str], encoder: OrdinalEncoder) -> np.ndarray:
    cat_values = x[cats].astype(str)
    cat_encoded = encoder.transform(cat_values)
    num_values = x[nums].astype("float64").to_numpy()
    return np.hstack([cat_encoded, num_values])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class MeanModel:
    """Global mean of the training fold (no features)."""

    def __init__(self) -> None:
        self.mean_ = 0.0

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "MeanModel":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.mean_, dtype="float64")

    def feature_importance(self) -> pd.DataFrame | None:
        return None


class RidgeModel:
    """One-hot categoricals + standardized numerics, Ridge on log1p(y)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "RidgeModel":
        self.cats_ = _cat_cols(x)
        self.nums_ = _num_cols(x, self.cats_)
        self.levels_ = _fit_levels(x, self.cats_)
        x = _apply_levels(x, self.cats_, self.levels_)

        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
        x_cat = self.encoder_.fit_transform(x[self.cats_].astype(str))
        self.scaler_ = StandardScaler()
        x_num = self.scaler_.fit_transform(x[self.nums_].astype("float64"))
        design = sparse.hstack([x_cat, sparse.csr_matrix(x_num)]).tocsr()

        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(design, np.log1p(y))
        return self

    def _design(self, x: pd.DataFrame) -> sparse.csr_matrix:
        x = _apply_levels(x, self.cats_, self.levels_)
        x_cat = self.encoder_.transform(x[self.cats_].astype(str))
        x_num = self.scaler_.transform(x[self.nums_].astype("float64"))
        return sparse.hstack([x_cat, sparse.csr_matrix(x_num)]).tocsr()

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.expm1(self.model_.predict(self._design(x)))

    def feature_importance(self) -> pd.DataFrame:
        names = list(self.encoder_.get_feature_names_out(self.cats_)) + list(self.nums_)
        coef = np.abs(np.asarray(self.model_.coef_).reshape(-1))
        return (
            pd.DataFrame({"feature": names, "importance": coef})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


class RFModel:
    """RandomForest on log1p(y); categoricals are ordinal-encoded."""

    def __init__(self, n_estimators: int = 300, min_samples_leaf: int = 3, n_jobs: int = -1, seed: int = 42):
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_jobs = int(n_jobs)
        self.seed = int(seed)

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "RFModel":
        self.cats_ = _cat_cols(x)
        self.nums_ = _num_cols(x, self.cats_)
        self.levels_ = _fit_levels(x, self.cats_)
        x = _apply_levels(x, self.cats_, self.levels_)
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64
        )
        self.encoder_.fit(x[self.cats_].astype(str))
        design = _matrix_rf(x, self.cats_, self.nums_, self.encoder_)
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )
        self.model_.fit(design, np.log1p(y))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x = _apply_levels(x, self.cats_, self.levels_)
        design = _matrix_rf(x, self.cats_, self.nums_, self.encoder_)
        return np.expm1(self.model_.predict(design))

    def feature_importance(self) -> pd.DataFrame:
        names = list(self.cats_) + list(self.nums_)
        return (
            pd.DataFrame({"feature": names, "importance": self.model_.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


class LGBMModel:
    """LightGBM wrapper; keeps consistent category dtypes across folds."""

    def __init__(self, params: dict[str, Any], log_target: bool = False, seed: int = 42):
        self.params = dict(params)
        self.log_target = bool(log_target)
        self.seed = int(seed)

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "LGBMModel":
        from lightgbm import LGBMRegressor

        self.cats_ = _cat_cols(x)
        self.levels_ = _fit_levels(x, self.cats_)
        x = _apply_levels(x, self.cats_, self.levels_)
        params = dict(self.params)
        params["random_state"] = self.seed
        self.model_ = LGBMRegressor(**params)
        target = np.log1p(y) if self.log_target else y
        self.model_.fit(x, target, categorical_feature=self.cats_)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x = _apply_levels(x, self.cats_, self.levels_)
        pred = self.model_.predict(x)
        return np.expm1(pred) if self.log_target else pred

    def feature_importance(self) -> pd.DataFrame:
        names = list(self.model_.feature_name_)
        gain = self.model_.booster_.feature_importance(importance_type="gain")
        split = self.model_.booster_.feature_importance(importance_type="split")
        return (
            pd.DataFrame({"feature": names, "importance": gain, "split": split})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def make_model(cfg: dict, model: str, seed: int):
    if model == "mean":
        return MeanModel()
    if model == "ridge":
        return RidgeModel(alpha=float(cfg.get("model", {}).get("ridge", {}).get("alpha", 1.0)))
    if model == "rf":
        p = cfg.get("model", {}).get("rf", {})
        return RFModel(
            n_estimators=int(p.get("n_estimators", 300)),
            min_samples_leaf=int(p.get("min_samples_leaf", 3)),
            n_jobs=int(p.get("n_jobs", -1)),
            seed=seed,
        )
    if model in ("lgbm", "lgbm_log"):
        params = model_params(cfg, model)
        params.pop("target_transform", None)
        return LGBMModel(params, log_target=(model == "lgbm_log"), seed=seed)
    raise ValueError(f"unknown model: {model}")


# ---------------------------------------------------------------------------
# Data / CV
# ---------------------------------------------------------------------------
def load_train(cfg: dict) -> pd.DataFrame:
    path = path_of(cfg, "processed_dir") / "train.parquet"
    return pd.read_parquet(path)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(pred, dtype="float64"), 0.0, None)
    return {
        "rmsle": rmsle(y, pred),
        "mae": mae(y, pred),
        "rmse": rmse(y, pred),
        "r2": r2(y, pred),
    }


def _append_exp_log(cfg: dict, row: dict[str, Any]) -> Path:
    path = path_of(cfg, "exp_log")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["run_id", "date", "model", "seed", "params_hash", "cv_rmsle", "cv_mae", "cv_rmse", "notes"]
    exists = path.exists() and path.stat().st_size > 0
    pd.DataFrame([{k: row[k] for k in columns}], columns=columns).to_csv(
        path, mode="a", header=not exists, index=False
    )
    return path


def run_cv(cfg: dict, model: str = "lgbm", seed: int = 42, log: bool = True) -> dict[str, Any]:
    """Run expanding-window time CV and persist OOF + experiment log."""
    if model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {model!r}")

    train = load_train(cfg)
    folds, payload = load_folds(cfg)
    target = cfg["data"]["target"]
    y = train[target].to_numpy(dtype="float64")
    oof = np.full(len(train), np.nan, dtype="float64")
    fold_metrics: list[dict[str, Any]] = []
    n_features = 0

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        builder = FeatureBuilder().fit(train.iloc[tr_idx])
        x_tr = builder.transform(train.iloc[tr_idx])
        x_va = builder.transform(train.iloc[va_idx])
        n_features = len(builder.feature_names)

        est = make_model(cfg, model, seed)
        est.fit(x_tr, y[tr_idx])
        pred = np.clip(np.asarray(est.predict(x_va), dtype="float64"), 0.0, None)
        oof[va_idx] = pred
        fm = _metrics(y[va_idx], pred)
        fm.update(
            {
                "fold": fold_id,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "valid_months": payload["folds"][fold_id].get("valid_months", []),
            }
        )
        fold_metrics.append(fm)
        if log:
            print(
                f"  fold {fold_id} | train={len(tr_idx):5d} valid={len(va_idx):4d} "
                f"| RMSLE={fm['rmsle']:.5f} MAE={fm['mae']:.3f} RMSE={fm['rmse']:.3f} R2={fm['r2']:.4f}"
            )

    # Expanding-window folds never validate the first 12 months (there is no
    # earlier training window), so OOF legitimately covers only 2012 onward.
    mask = ~np.isnan(oof)
    if not mask.any():
        raise RuntimeError("OOF is empty; no validation predictions were produced")
    n_oof = int(mask.sum())
    coverage = n_oof / len(train)
    pooled = _metrics(y[mask], oof[mask])
    phash = params_hash(cfg, model)
    run_id = _make_run_id(model, seed, phash)
    oof_df = pd.DataFrame(
        {
            "row": np.arange(len(train), dtype=int)[mask],
            "y": y[mask],
            "pred": oof[mask],
        }
    )

    oof_dir = path_of(cfg, "oof_dir")
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{run_id}.parquet"
    oof_df.to_parquet(oof_path, index=False)

    notes = (
        f"5-fold expanding-window time CV; pooled OOF on {n_oof}/{len(train)} rows "
        f"({coverage:.1%}, 2012-01 onward); "
        f"target={target}; objective={model_params(cfg, model).get('objective', 'n/a')}; "
        f"n_features={n_features}"
    )
    _append_exp_log(
        cfg,
        {
            "run_id": run_id,
            "date": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "seed": int(seed),
            "params_hash": phash,
            "cv_rmsle": pooled["rmsle"],
            "cv_mae": pooled["mae"],
            "cv_rmse": pooled["rmse"],
            "notes": notes,
        },
    )

    if log:
        print(
            f"  CV pooled | RMSLE={pooled['rmsle']:.5f} MAE={pooled['mae']:.3f} "
            f"RMSE={pooled['rmse']:.3f} R2={pooled['r2']:.4f} "
            f"| OOF rows={n_oof}/{len(train)} ({coverage:.1%})"
        )
        print(f"  run_id={run_id}")
        print(f"  OOF -> {oof_path}")

    return {
        "run_id": run_id,
        "model": model,
        "seed": int(seed),
        "params_hash": phash,
        "n_features": n_features,
        "n_oof": n_oof,
        "oof_coverage": coverage,
        "cv_rmsle": pooled["rmsle"],
        "cv_mae": pooled["mae"],
        "cv_rmse": pooled["rmse"],
        "cv_r2": pooled["r2"],
        "fold_metrics": fold_metrics,
        "oof": oof_df,
        "oof_path": str(oof_path),
    }


def compare_models(cfg: dict, models: tuple[str, ...] | list[str] = COMPARE_MODELS, seed: int = 42) -> pd.DataFrame:
    """Run CV for several models, write comparison CSV and figures."""
    rows = []
    for model in models:
        print(f"[{model}]")
        res = run_cv(cfg, model=model, seed=seed)
        row: dict[str, Any] = {
            "model": model,
            "seed": seed,
            "run_id": res["run_id"],
            "params_hash": res["params_hash"],
            "n_features": res["n_features"],
            "n_oof": res["n_oof"],
            "oof_coverage": res["oof_coverage"],
            "cv_rmsle": res["cv_rmsle"],
            "cv_mae": res["cv_mae"],
            "cv_rmse": res["cv_rmse"],
            "cv_r2": res["cv_r2"],
            "mean_fold_rmsle": float(np.mean([f["rmsle"] for f in res["fold_metrics"]])),
        }
        for i, fm in enumerate(res["fold_metrics"]):
            row[f"fold_{i}_rmsle"] = fm["rmsle"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("cv_rmsle").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    out = path_of(cfg, "reports") / "p2_model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    fig = path_of(cfg, "figures") / "p2_model_comparison.png"
    plot_comparison(df, fig)
    plot_fold_rmsle(df, path_of(cfg, "figures") / "p2_fold_rmsle.png")

    print("\nModel comparison (sorted by pooled OOF RMSLE):")
    print(
        df[["rank", "model", "cv_rmsle", "cv_mae", "cv_rmse", "cv_r2"]].to_string(
            index=False, float_format=lambda v: f"{v:.5f}"
        )
    )
    print(f"\nComparison -> {out}")
    print(f"Figure     -> {fig}")
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_comparison(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [("cv_rmsle", "RMSLE (lower is better)"), ("cv_mae", "MAE"), ("cv_rmse", "RMSE")]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(df)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (metric, title) in zip(axes, metrics):
        bars = ax.bar(df["model"], df[metric], color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel(metric.replace("cv_", "").upper())
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}" if metric == "cv_rmsle" else f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Bike Sharing Demand - 5-fold expanding-window time CV")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_fold_rmsle(df: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    fold_cols = [c for c in df.columns if c.startswith("fold_") and c.endswith("_rmsle")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for _, row in df.iterrows():
        ax.plot(range(len(fold_cols)), [row[c] for c in fold_cols], marker="o", label=row["model"])
    ax.set_xticks(range(len(fold_cols)))
    ax.set_xticklabels([c.replace("fold_", "fold ").replace("_rmsle", "") for c in fold_cols])
    ax.set_xlabel("Validation fold (expanding window)")
    ax.set_ylabel("RMSLE")
    ax.set_title("Per-fold RMSLE by model")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Bike Sharing Demand time-CV training (P2)")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="lgbm", choices=list(MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", action="store_true", help="run all comparison models and write p2_model_comparison.csv")
    parser.add_argument("--models", default=",".join(COMPARE_MODELS), help="comma-separated models for --compare")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.compare:
        models = tuple(m.strip() for m in args.models.split(",") if m.strip())
        compare_models(cfg, models=models, seed=args.seed)
    else:
        run_cv(cfg, model=args.model, seed=args.seed)


if __name__ == "__main__":
    main()
