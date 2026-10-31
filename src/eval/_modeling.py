"""Small modelling helpers shared by the P3 fallback and SHAP scripts.

These are *evaluation-side* utilities.  They do not modify P2's training code
and are only used when the final P2 artifacts are unavailable (smoke test) or
when the interpretation script needs a full-train LightGBM for TreeSHAP.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.features.build import CAT_COLS, FeatureBuilder


def lgbm_params(cfg: dict, quick: bool = False, seed: int = 42, log_target: bool = True) -> dict:
    """Effective LightGBM params.

    ``log_target=True`` trains L2 on log1p(count) (P2's ``lgbm_log`` recipe).
    ``log_target=False`` keeps the config objective (Poisson) on raw counts
    (P2's ``lgbm`` recipe, which is the current final model).
    """
    params = dict(cfg.get("model", {}).get("lgbm", {}))
    if log_target:
        params["objective"] = "regression"
    else:
        params["objective"] = params.get("objective", "poisson")
    n_jobs = int(params.get("n_jobs", -1))
    if n_jobs < 0:
        # A very large OpenMP thread count can deadlock/hang on small data in
        # some LightGBM builds; cap evaluation-side training for robustness.
        n_jobs = min(4, os.cpu_count() or 1)
    params.update(
        random_state=int(seed),
        n_jobs=n_jobs,
        verbose=-1,
    )
    if quick:
        params.update(
            n_estimators=250,
            learning_rate=0.07,
            num_leaves=31,
            min_child_samples=20,
        )
    return params


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns (e.g. ``hour_x_workingday``) to pandas category."""
    out = X.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).astype("category")
    return out


def align_categories(reference: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give both frames the same pandas category dictionaries (LightGBM-safe)."""
    ref = prepare_features(reference)
    oth = prepare_features(other)
    for col in ref.columns:
        if col not in oth.columns:
            continue
        if not isinstance(ref[col].dtype, pd.CategoricalDtype):
            continue
        if not isinstance(oth[col].dtype, pd.CategoricalDtype):
            continue
        cats = sorted(set(ref[col].cat.categories) | set(oth[col].cat.categories))
        ref[col] = ref[col].cat.set_categories(cats)
        oth[col] = oth[col].cat.set_categories(cats)
    return ref, oth


def fit_lgbm(
    X: pd.DataFrame,
    y,
    cfg: dict,
    quick: bool = False,
    seed: int = 42,
    log_target: bool = True,
):
    """Fit a LightGBM with either log1p+L2 or the configured (Poisson) objective."""
    from lightgbm import LGBMRegressor

    X = prepare_features(X)
    params = lgbm_params(cfg, quick=quick, seed=seed, log_target=log_target)
    target = np.log1p(np.asarray(y, dtype="float64")) if log_target else np.asarray(y, dtype="float64")
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    model = LGBMRegressor(**params)
    model.fit(X, target, categorical_feature=cats)
    return model


def fit_lgbm_log(X: pd.DataFrame, y, cfg: dict, quick: bool = False, seed: int = 42):
    """Backward-compatible helper used by the smoke baseline."""
    return fit_lgbm(X, y, cfg, quick=quick, seed=seed, log_target=True)


def predict_model(model, X: pd.DataFrame, log_target: bool) -> np.ndarray:
    pred = np.asarray(model.predict(X), dtype="float64")
    if log_target:
        pred = np.expm1(pred)
    return np.clip(pred, 0.0, None)


def predict_counts(model, X: pd.DataFrame) -> np.ndarray:
    """Backward-compatible helper for log-target models."""
    return predict_model(model, X, log_target=True)


def quick_baseline_oof(cfg: dict, seed: int = 42, log: bool = True) -> tuple[pd.DataFrame, dict]:
    """Produce a temporary, honest time-CV OOF for smoke tests only.

    This deliberately uses a small LightGBM and the frozen folds.  It is *not*
    P2's model and must never be written to ``reports/oof/final_oof.parquet``.
    """
    from src.common.config import path_of
    from src.common.split import load_folds

    train = pd.read_parquet(path_of(cfg, "processed_dir") / "train.parquet")
    folds, _ = load_folds(cfg)
    parts: list[pd.DataFrame] = []
    fold_scores: list[dict] = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        tr = train.iloc[tr_idx].reset_index(drop=True)
        va = train.iloc[va_idx].reset_index(drop=True)
        builder = FeatureBuilder(use_target_aggregates=True).fit(tr)
        xtr = builder.transform(tr)
        xva = builder.transform(va)
        xtr, xva = align_categories(xtr, xva)
        model = fit_lgbm_log(xtr, tr["count"], cfg, quick=True, seed=seed + i)
        pred = predict_counts(model, xva)
        y = va["count"].to_numpy(dtype="float64")
        from src.common.metrics import rmsle

        fold_scores.append({"fold": i, "n": int(len(y)), "rmsle": float(rmsle(y, pred))})
        parts.append(
            pd.DataFrame(
                {
                    "row": np.asarray(va_idx, dtype=int),
                    "y": y,
                    "pred": pred,
                }
            )
        )
        if log:
            print(f"[smoke] fold {i}: n={len(y):5d} rmsle={fold_scores[-1]['rmsle']:.5f}")
    oof = pd.concat(parts, ignore_index=True).sort_values("row").reset_index(drop=True)
    return oof, {"folds": fold_scores, "note": "temporary quick baseline; not P2 final OOF"}


def config_for_eval(path: str) -> dict:
    return load_config(path)
