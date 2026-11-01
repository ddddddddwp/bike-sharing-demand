#!/usr/bin/env python3
"""Horizontal extension experiments for Bike Sharing Demand.

A. hyperparameter sensitivity + convergence
B. feature / preprocessing ablation
C. model efficiency + lightweight models
D. temporal drift: static vs expanding vs sliding window
E. multi-seed stability + bootstrap CI + paired test

Usage:
    python3 scripts/run_extensions.py --experiment all
    python3 scripts/run_extensions.py --experiment a
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

from src.common.config import load_config, path_of  # noqa: E402
from src.common.metrics import mae, rmse, rmsle  # noqa: E402
from src.common.split import load_folds  # noqa: E402
from src.features.build import FeatureBuilder  # noqa: E402
from src.models.train import LGBMModel, RFModel, RidgeModel, model_params, _apply_levels, _cat_cols, _fit_levels  # noqa: E402

EXT = ROOT / "reports" / "extension"
FIG = ROOT / "reports" / "figures"
EXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_train(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(path_of(cfg, "processed_dir") / "train.parquet")


def lgbm_params(cfg: dict, **overrides) -> dict:
    p = dict(model_params(cfg, "lgbm"))
    p.pop("target_transform", None)
    p.update(overrides)
    return p


def _metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    return {"rmsle": rmsle(y, p), "mae": mae(y, p), "rmse": rmse(y, p)}


def _drop_keep(x: pd.DataFrame, drop: list[str] | None, keep: list[str] | None) -> pd.DataFrame:
    if drop:
        x = x.drop(columns=[c for c in drop if c in x.columns])
    if keep is not None:
        x = x[[c for c in keep if c in x.columns]]
    return x


def cv_lgbm(
    cfg: dict,
    *,
    seed: int = 42,
    params: dict | None = None,
    drop_cols: list[str] | None = None,
    keep_cols: list[str] | None = None,
    log_target: bool = False,
) -> dict:
    tr = load_train(cfg)
    folds, _ = load_folds(cfg)
    y = tr["count"].to_numpy(float)
    oof = np.full(len(tr), np.nan)
    train_seconds = 0.0
    n_features = 0
    for tr_idx, va_idx in folds:
        b = FeatureBuilder().fit(tr.iloc[tr_idx])
        xt = _drop_keep(b.transform(tr.iloc[tr_idx]), drop_cols, keep_cols)
        xv = _drop_keep(b.transform(tr.iloc[va_idx]), drop_cols, keep_cols)
        n_features = xt.shape[1]
        p = dict(params or lgbm_params(cfg))
        if log_target:
            p["objective"] = "regression"
        m = LGBMModel(p, log_target=log_target, seed=seed)
        t0 = time.perf_counter()
        m.fit(xt, y[tr_idx])
        train_seconds += time.perf_counter() - t0
        oof[va_idx] = np.clip(m.predict(xv), 0.0, None)
    mask = ~np.isnan(oof)
    return {
        "oof": oof,
        "mask": mask,
        "metrics": _metrics(y[mask], oof[mask]),
        "train_seconds": train_seconds,
        "n_features": n_features,
    }


def _rmsle_eval(y_true, y_pred):
    p = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return "rmsle", float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y_true)) ** 2))), False


def _fit_es(params: dict, xt, yt, xv, yv, seed: int):
    cats = _cat_cols(xt)
    levels = _fit_levels(xt, cats)
    xt = _apply_levels(xt, cats, levels)
    xv = _apply_levels(xv, cats, levels)
    p = dict(params)
    p["n_estimators"] = int(p.get("n_estimators", 3000))
    p["random_state"] = seed
    m = lgb.LGBMRegressor(**p)
    m.fit(
        xt,
        yt,
        eval_set=[(xt, yt), (xv, yv)],
        eval_names=["train", "valid"],
        eval_metric=_rmsle_eval,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return m, xv


# ---------------------------------------------------------------------------
# A. hyperparameter sensitivity + convergence
# ---------------------------------------------------------------------------
def run_hyperparam(cfg: dict) -> pd.DataFrame:
    """Fixed-800 comparison + an early-stopping convergence diagnostic.

    The fixed-800 grid is comparable to P2's final model. Early stopping is
    reported separately because it stops at a different iteration budget and is
    not directly comparable to the fixed-budget CV.
    """
    tr = load_train(cfg)
    folds, _ = load_folds(cfg)
    y = tr["count"].to_numpy(float)
    base = lgbm_params(cfg)
    configs = []
    for v in (0.01, 0.03, 0.05, 0.1):
        configs.append(("learning_rate", v, {**base, "learning_rate": v}))
    for v in (15, 31, 63, 127):
        configs.append(("num_leaves", v, {**base, "num_leaves": v}))
    for v in (10, 30, 50):
        configs.append(("min_child_samples", v, {**base, "min_child_samples": v}))

    rows = []
    for name, value, params in configs:
        res = cv_lgbm(cfg, params=params)
        met = res["metrics"]
        rows.append(
            {
                "param": name,
                "value": value,
                "n_features": res["n_features"],
                "cv_rmsle": met["rmsle"],
                "cv_mae": met["mae"],
                "cv_rmse": met["rmse"],
                "train_seconds": res["train_seconds"],
            }
        )
        print(f"[A] {name}={value}: RMSLE={met['rmsle']:.5f}")

    df = pd.DataFrame(rows)
    df.to_csv(EXT / "hyperparam.csv", index=False)

    # convergence / early stopping diagnostic on fold 0
    tr_idx, va_idx = folds[0]
    b = FeatureBuilder().fit(tr.iloc[tr_idx])
    xt, xv = b.transform(tr.iloc[tr_idx]), b.transform(tr.iloc[va_idx])
    m, xv_aligned = _fit_es({**base, "n_estimators": 3000, "learning_rate": 0.05},
                            xt, y[tr_idx], xv, y[va_idx], seed=42)
    ev = m.evals_result_
    conv = pd.DataFrame(
        {
            "iter": np.arange(1, len(ev["valid"]["rmsle"]) + 1),
            "train_rmsle": ev["train"]["rmsle"],
            "valid_rmsle": ev["valid"]["rmsle"],
        }
    )
    conv.to_csv(EXT / "convergence.csv", index=False)
    es_met = _metrics(y[va_idx], m.predict(xv_aligned))
    fixed_met = _metrics(y[va_idx], cv_lgbm(cfg, params=base)["oof"][va_idx])
    pd.DataFrame(
        [
            {
                "setting": "early_stopping_3000",
                "best_iter": int(m.best_iteration_),
                "stop_iter": int(m.booster_.current_iteration()),
                "valid_rmsle": es_met["rmsle"],
            },
            {
                "setting": "fixed_800",
                "best_iter": 800,
                "stop_iter": 800,
                "valid_rmsle": fixed_met["rmsle"],
            },
        ]
    ).to_csv(EXT / "convergence_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, name in zip(axes, ["learning_rate", "num_leaves", "min_child_samples"]):
        d = df[df["param"] == name]
        ax.plot(d["value"].astype(str), d["cv_rmsle"], "o-", color="#2b6cb0")
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("CV RMSLE")
        ax.grid(alpha=0.25)
        for _, r in d.iterrows():
            ax.annotate(f"{r['cv_rmsle']:.4f}", (str(r["value"]), r["cv_rmsle"]),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "ext_hyperparam.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(conv["iter"], conv["train_rmsle"], label="train", color="#2b6cb0", lw=1.2)
    ax.plot(conv["iter"], conv["valid_rmsle"], label="valid", color="#c53030", lw=1.2)
    ax.axvline(m.best_iteration_, color="#718096", ls="--", lw=1,
               label=f"best_iter={m.best_iteration_}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("RMSLE")
    ax.set_title("LightGBM convergence (fold 0, lr=0.05, early stopping=50)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "ext_convergence.png", dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# B. feature / preprocessing ablation
# ---------------------------------------------------------------------------
def run_feature_ablation(cfg: dict) -> pd.DataFrame:
    tr = load_train(cfg)
    b = FeatureBuilder().fit(tr)
    all_cols = list(b.transform(tr.head(5)).columns)
    target_cols = [c for c in all_cols if c.endswith("_mean")]
    cyclical = [c for c in all_cols if c.endswith("_sin") or c.endswith("_cos")]
    interactions = [c for c in ("temp_x_season", "humidity_x_weather", "hour_x_workingday") if c in all_cols]
    variants = [
        ("full", None, None, False),
        ("no_target_agg", target_cols, None, False),
        ("no_cyclical", cyclical, None, False),
        ("no_interactions", interactions, None, False),
        ("base_calendar_weather", target_cols + cyclical + interactions, None, False),
        ("no_year", ["year"], None, False),
        ("no_day", ["day"], None, False),
        ("log1p_target", None, None, True),
    ]
    rows = []
    for name, drop, keep, log_target in variants:
        res = cv_lgbm(cfg, params=lgbm_params(cfg), drop_cols=drop, keep_cols=keep, log_target=log_target)
        m = res["metrics"]
        rows.append(
            {
                "variant": name,
                "n_features": res["n_features"],
                "cv_rmsle": m["rmsle"],
                "cv_mae": m["mae"],
                "cv_rmse": m["rmse"],
                "train_seconds": res["train_seconds"],
            }
        )
        print(f"[B] {name}: RMSLE={m['rmsle']:.5f} n_feat={res['n_features']}")
    df = pd.DataFrame(rows).sort_values("cv_rmsle").reset_index(drop=True)
    df.to_csv(EXT / "feature_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    d = df.sort_values("cv_rmsle")
    ax.barh(d["variant"], d["cv_rmsle"], color="#2b6cb0")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r["cv_rmsle"], i, f" {r['cv_rmsle']:.4f} ({int(r['n_features'])}f)", va="center", fontsize=7)
    ax.set_xlabel("CV RMSLE")
    ax.set_title("Feature / preprocessing ablation (time_v2)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "ext_feature_ablation.png", dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# C. model efficiency
# ---------------------------------------------------------------------------
def _model_bytes(model) -> int:
    if hasattr(model, "booster_"):
        return len(model.booster_.model_to_string().encode("utf-8"))
    try:
        return len(pickle.dumps(model))
    except Exception:
        return 0


def _latency_ms(model, xv: pd.DataFrame, n: int = 1000, repeat: int = 10) -> float:
    x = xv.head(min(n, len(xv)))
    model.predict(x)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        model.predict(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def run_efficiency(cfg: dict) -> pd.DataFrame:
    tr = load_train(cfg)
    folds, _ = load_folds(cfg)
    y = tr["count"].to_numpy(float)

    def eval_model(kind: str, params: dict | None = None, topk: int | None = None, log_target: bool = False):
        oof = np.full(len(tr), np.nan)
        secs = 0.0
        sizes, lat = [], []
        nfeat = 0
        for tr_idx, va_idx in folds:
            b = FeatureBuilder().fit(tr.iloc[tr_idx])
            xt, xv = b.transform(tr.iloc[tr_idx]), b.transform(tr.iloc[va_idx])
            if topk is not None:
                selector = LGBMModel(lgbm_params(cfg, n_estimators=200), seed=42)
                selector.fit(xt, y[tr_idx])
                imp = selector.feature_importance()
                keep = imp.head(topk)["feature"].tolist()
                xt, xv = xt[keep], xv[keep]
            nfeat = xt.shape[1]
            if kind == "lgbm":
                m = LGBMModel(params or lgbm_params(cfg), log_target=log_target, seed=42)
            elif kind == "ridge":
                m = RidgeModel(alpha=float(cfg.get("model", {}).get("ridge", {}).get("alpha", 1.0)))
            elif kind == "rf":
                p = cfg.get("model", {}).get("rf", {})
                m = RFModel(n_estimators=int(p.get("n_estimators", 300)),
                            min_samples_leaf=int(p.get("min_samples_leaf", 3)),
                            n_jobs=int(p.get("n_jobs", -1)), seed=42)
            else:
                raise ValueError(kind)
            t0 = time.perf_counter()
            m.fit(xt, y[tr_idx])
            secs += time.perf_counter() - t0
            oof[va_idx] = np.clip(m.predict(xv), 0.0, None)
            sizes.append(_model_bytes(m))
            lat.append(_latency_ms(m, xv))
        mask = ~np.isnan(oof)
        met = _metrics(y[mask], oof[mask])
        return {
            "model": kind,
            "n_features": nfeat,
            "cv_rmsle": met["rmsle"],
            "cv_mae": met["mae"],
            "cv_rmse": met["rmse"],
            "train_seconds": secs,
            "latency_ms_1k": float(np.median(lat)),
            "model_kb": float(np.median(sizes) / 1024.0),
        }

    rows = [
        eval_model("lgbm"),
        eval_model("lgbm", params=lgbm_params(cfg, num_leaves=15, n_estimators=200), topk=None),
        eval_model("lgbm", topk=10),
        eval_model("lgbm", topk=20),
        eval_model("ridge"),
        eval_model("rf"),
    ]
    names = ["lgbm_full", "lgbm_small", "lgbm_top10", "lgbm_top20", "ridge", "rf"]
    for r, n in zip(rows, names):
        r["model"] = n
        print(f"[C] {n}: RMSLE={r['cv_rmsle']:.5f} lat={r['latency_ms_1k']:.2f}ms size={r['model_kb']:.1f}KB")
    df = pd.DataFrame(rows).sort_values("cv_rmsle").reset_index(drop=True)
    df.to_csv(EXT / "efficiency.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, xcol, xlabel in [(axes[0], "latency_ms_1k", "inference latency (ms / 1k rows)"),
                             (axes[1], "model_kb", "model size (KB)")]:
        ax.scatter(df[xcol], df["cv_rmsle"], s=50, color="#2b6cb0")
        for _, r in df.iterrows():
            ax.annotate(r["model"], (r[xcol], r["cv_rmsle"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CV RMSLE")
        ax.grid(alpha=0.25)
    fig.suptitle("Model efficiency Pareto (same time_v2 folds, CPU, median latency)")
    fig.tight_layout()
    fig.savefig(FIG / "ext_efficiency.png", dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# D. temporal drift
# ---------------------------------------------------------------------------
def run_temporal(cfg: dict) -> pd.DataFrame:
    tr = load_train(cfg)
    dt = pd.to_datetime(tr["datetime"])
    ym = dt.dt.to_period("M").astype(str)
    months = [m for m in sorted(ym.unique()) if m.startswith("2012-")]
    y = tr["count"].to_numpy(float)
    rows = []

    def predict_month(train_mask, month):
        tr_idx = np.where(train_mask)[0]
        va_idx = np.where(ym == month)[0]
        b = FeatureBuilder().fit(tr.iloc[tr_idx])
        xt, xv = b.transform(tr.iloc[tr_idx]), b.transform(tr.iloc[va_idx])
        m = LGBMModel(lgbm_params(cfg), seed=42)
        m.fit(xt, y[tr_idx])
        p = np.clip(m.predict(xv), 0.0, None)
        return _metrics(y[va_idx], p), len(tr_idx), len(va_idx)

    for month in months:
        start = pd.Period(month).start_time
        # static: 2011 only
        static_mask = (dt < pd.Timestamp("2012-01-01")).to_numpy()
        static_met, ntr, nva = predict_month(static_mask, month)
        # expanding: all data before month
        exp_mask = (dt < start).to_numpy()
        exp_met, _, _ = predict_month(exp_mask, month)
        # sliding 12 months
        slide_start = start - pd.DateOffset(months=12)
        slide_mask = ((dt >= slide_start) & (dt < start)).to_numpy()
        slide_met, _, _ = predict_month(slide_mask, month)
        rows.append({
            "month": month,
            "n_train_static": ntr,
            "n_valid": nva,
            "static_rmsle": static_met["rmsle"],
            "expanding_rmsle": exp_met["rmsle"],
            "sliding12_rmsle": slide_met["rmsle"],
            "static_mae": static_met["mae"],
            "expanding_mae": exp_met["mae"],
            "sliding12_mae": slide_met["mae"],
        })
        print(f"[D] {month}: static={static_met['rmsle']:.4f} expanding={exp_met['rmsle']:.4f} sliding12={slide_met['rmsle']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(EXT / "temporal_shift.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.0))
    for col, label in [("static_rmsle", "static (2011 only)"),
                       ("expanding_rmsle", "expanding (all past)"),
                       ("sliding12_rmsle", "sliding 12 months")]:
        ax.plot(df["month"], df[col], "o-", label=label)
    ax.set_xlabel("validation month (2012)")
    ax.set_ylabel("RMSLE")
    ax.set_title("Temporal drift: static vs expanding vs sliding window")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "ext_temporal_shift.png", dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# E. multi-seed stability + bootstrap CI + paired test
# ---------------------------------------------------------------------------
def run_stability(cfg: dict) -> pd.DataFrame:
    tr = load_train(cfg)
    folds, _ = load_folds(cfg)
    y = tr["count"].to_numpy(float)
    rows = []
    oofs = {}
    for seed in (42, 2026, 7):
        res = cv_lgbm(cfg, seed=seed, params=lgbm_params(cfg))
        oofs[seed] = res["oof"]
        mask = res["mask"]
        fold_rmsle = []
        for tr_idx, va_idx in folds:
            fold_rmsle.append(rmsle(y[va_idx], res["oof"][va_idx]))
        rows.append({
            "seed": seed,
            "cv_rmsle": res["metrics"]["rmsle"],
            "cv_mae": res["metrics"]["mae"],
            "cv_rmse": res["metrics"]["rmse"],
            "fold_rmsle_mean": float(np.mean(fold_rmsle)),
            "fold_rmsle_std": float(np.std(fold_rmsle)),
            "fold_rmsle": json.dumps([round(float(x), 6) for x in fold_rmsle]),
            "n_oof": int(mask.sum()),
        })
        print(f"[E] seed={seed}: RMSLE={res['metrics']['rmsle']:.5f}")

    df = pd.DataFrame(rows)
    df.to_csv(EXT / "stability.csv", index=False)

    # bootstrap CI on seed 42 OOF
    oof = oofs[42]
    mask = ~np.isnan(oof)
    yy, pp = y[mask], oof[mask]
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(yy), len(yy))
        boots.append(rmsle(yy[idx], pp[idx]))
    ci = {
        "seed": 42,
        "rmsle": rmsle(yy, pp),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n_boot": 1000,
    }

    # paired test: lgbm vs lgbm_log fold RMSLE from P2 comparison
    paired = {"status": "missing"}
    comp_path = path_of(cfg, "reports") / "p2_model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path).set_index("model")
        if {"lgbm", "lgbm_log"} <= set(comp.index):
            a = [comp.loc["lgbm", f"fold_{i}_rmsle"] for i in range(5)]
            b = [comp.loc["lgbm_log", f"fold_{i}_rmsle"] for i in range(5)]
            try:
                stat, p = stats.wilcoxon(a, b)
                paired = {"models": ["lgbm", "lgbm_log"], "stat": float(stat), "p_value": float(p), "n": 5}
            except Exception as e:
                paired = {"status": f"failed: {e}"}
    ci["paired_test"] = paired
    (EXT / "stability_ci.json").write_text(json.dumps(ci, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(df["seed"].astype(str), df["cv_rmsle"], "o-", color="#2b6cb0", label="OOF RMSLE")
    for _, r in df.iterrows():
        ax.annotate(f"{r['cv_rmsle']:.5f}", (str(r["seed"]), r["cv_rmsle"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(df["cv_rmsle"].mean(), color="#c53030", ls="--", label=f"mean={df['cv_rmsle'].mean():.5f}")
    ax.set_xlabel("seed")
    ax.set_ylabel("OOF RMSLE")
    ax.set_title("Multi-seed stability (time_v2, LightGBM Poisson)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "ext_stability.png", dpi=200)
    plt.close(fig)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="all", choices=["all", "a", "b", "c", "d", "e"])
    args = ap.parse_args()
    cfg = load_config("configs/base.yaml")
    if args.experiment in ("all", "a"):
        run_hyperparam(cfg)
    if args.experiment in ("all", "b"):
        run_feature_ablation(cfg)
    if args.experiment in ("all", "c"):
        run_efficiency(cfg)
    if args.experiment in ("all", "d"):
        run_temporal(cfg)
    if args.experiment in ("all", "e"):
        run_stability(cfg)
    print(f"[ext] outputs -> {EXT}")


if __name__ == "__main__":
    main()
