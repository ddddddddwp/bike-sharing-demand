#!/usr/bin/env python3
"""Model / algorithm breadth benchmark on the same time_v2 folds.

Compares:
- LightGBM CPU (native categorical)
- XGBoost CPU / CUDA (one-hot), depth 4/6/8, Poisson vs log-target
- CatBoost CPU / GPU (one-hot), Poisson
- Ridge / RandomForest CPU

Usage:
    python3 scripts/model_zoo_benchmark.py
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.common.config import load_config, path_of  # noqa: E402
from src.common.metrics import mae, rmse, rmsle  # noqa: E402
from src.common.split import load_folds  # noqa: E402
from src.features.build import FeatureBuilder  # noqa: E402
from src.models.train import LGBMModel, RFModel, RidgeModel, model_params  # noqa: E402

EXT = ROOT / "reports" / "extension"
FIG = ROOT / "reports" / "figures"
EXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def one_hot(xt: pd.DataFrame, xv: pd.DataFrame):
    cats = [c for c in xt.columns if isinstance(xt[c].dtype, pd.CategoricalDtype) or xt[c].dtype == object]
    xt = pd.get_dummies(xt, columns=cats, dummy_na=False)
    xv = pd.get_dummies(xv, columns=cats, dummy_na=False)
    xv = xv.reindex(columns=xt.columns, fill_value=0)
    return xt.astype("float32"), xv.astype("float32")


def model_size_kb(model) -> float:
    try:
        return len(pickle.dumps(model)) / 1024.0
    except Exception:
        tmp = Path("/tmp/opencode/_model_size.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(tmp))
        size = tmp.stat().st_size / 1024.0
        tmp.unlink(missing_ok=True)
        return size


def latency_ms(model, xv: pd.DataFrame, n: int = 1000, repeat: int = 8) -> float:
    x = xv.head(min(n, len(xv)))
    model.predict(x)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        model.predict(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def run(cfg: dict) -> pd.DataFrame:
    import catboost as cb
    import xgboost as xgb

    tr = pd.read_parquet(path_of(cfg, "processed_dir") / "train.parquet")
    folds, _ = load_folds(cfg)
    y = tr["count"].to_numpy(float)

    def eval_family(name: str, kind: str, **kw) -> dict:
        oof = np.full(len(tr), np.nan)
        secs = 0.0
        sizes, lat = [], []
        nfeat = 0
        for tr_idx, va_idx in folds:
            b = FeatureBuilder().fit(tr.iloc[tr_idx])
            xt, xv = b.transform(tr.iloc[tr_idx]), b.transform(tr.iloc[va_idx])
            if kind in ("xgb", "cat"):
                xt, xv = one_hot(xt, xv)
            nfeat = xt.shape[1]
            if kind == "lgbm":
                model = LGBMModel(kw.get("params") or model_params(cfg, "lgbm"), seed=42)
            elif kind == "ridge":
                model = RidgeModel(alpha=float(cfg.get("model", {}).get("ridge", {}).get("alpha", 1.0)))
            elif kind == "rf":
                p = cfg.get("model", {}).get("rf", {})
                model = RFModel(n_estimators=int(p.get("n_estimators", 300)),
                                min_samples_leaf=int(p.get("min_samples_leaf", 3)),
                                n_jobs=int(p.get("n_jobs", -1)), seed=42)
            elif kind == "xgb":
                model = xgb.XGBRegressor(
                    n_estimators=800,
                    learning_rate=0.05,
                    max_depth=int(kw.get("depth", 6)),
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective=kw.get("objective", "count:poisson"),
                    tree_method="hist",
                    device=kw.get("device", "cpu"),
                    n_jobs=4 if kw.get("device", "cpu") == "cpu" else 1,
                    random_state=42,
                )
            elif kind == "cat":
                model = cb.CatBoostRegressor(
                    iterations=800,
                    learning_rate=0.05,
                    depth=int(kw.get("depth", 6)),
                    l2_leaf_reg=3.0,
                    loss_function=kw.get("loss", "Poisson"),
                    task_type=kw.get("task_type", "CPU"),
                    devices="0" if kw.get("task_type") == "GPU" else None,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=4,
                )
            else:
                raise ValueError(kind)
            t0 = time.perf_counter()
            if kind == "xgb":
                target = np.log1p(y[tr_idx]) if kw.get("log_target") else y[tr_idx]
                model.fit(xt, target, eval_set=[(xv, np.log1p(y[va_idx]) if kw.get("log_target") else y[va_idx])],
                          verbose=False)
                p = np.expm1(model.predict(xv)) if kw.get("log_target") else model.predict(xv)
            elif kind == "cat":
                model.fit(xt, y[tr_idx], eval_set=(xv, y[va_idx]))
                p = model.predict(xv)
            else:
                model.fit(xt, y[tr_idx])
                p = model.predict(xv)
            secs += time.perf_counter() - t0
            oof[va_idx] = np.clip(p, 0.0, None)
            sizes.append(model_size_kb(model))
            lat.append(latency_ms(model, xv))
        mask = ~np.isnan(oof)
        return {
            "model": name,
            "n_features": nfeat,
            "cv_rmsle": rmsle(y[mask], oof[mask]),
            "cv_mae": mae(y[mask], oof[mask]),
            "cv_rmse": rmse(y[mask], oof[mask]),
            "train_seconds": secs,
            "latency_ms_1k": float(np.median(lat)),
            "model_kb": float(np.median(sizes)),
        }

    specs = [
        ("lgbm_cpu", "lgbm", {}),
        ("xgb_cpu", "xgb", {"device": "cpu"}),
        ("xgb_cuda", "xgb", {"device": "cuda"}),
        ("xgb_cuda_depth4", "xgb", {"device": "cuda", "depth": 4}),
        ("xgb_cuda_depth8", "xgb", {"device": "cuda", "depth": 8}),
        ("xgb_cuda_log", "xgb", {"device": "cuda", "objective": "reg:squarederror", "log_target": True}),
        ("cat_cpu", "cat", {"task_type": "CPU"}),
        ("cat_gpu", "cat", {"task_type": "GPU"}),
        ("cat_gpu_depth4", "cat", {"task_type": "GPU", "depth": 4}),
        ("ridge", "ridge", {}),
        ("rf", "rf", {}),
    ]
    rows = []
    for name, kind, kw in specs:
        print(f"[zoo] {name} ...")
        try:
            r = eval_family(name, kind, **kw)
            r["status"] = "ok"
        except Exception as e:
            print(f"[zoo] {name} FAILED: {type(e).__name__}: {e}")
            r = {"model": name, "status": f"failed: {type(e).__name__}: {e}"}
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(EXT / "model_zoo.csv", index=False)

    ok = df[df["status"] == "ok"].copy().sort_values("cv_rmsle")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].scatter(ok["train_seconds"], ok["cv_rmsle"], s=45, color="#2b6cb0")
    for _, r in ok.iterrows():
        axes[0].annotate(r["model"], (r["train_seconds"], r["cv_rmsle"]),
                         textcoords="offset points", xytext=(5, 4), fontsize=7)
    axes[0].set_xlabel("training seconds (5 folds)")
    axes[0].set_ylabel("CV RMSLE")
    axes[0].set_title("Accuracy vs training cost")
    axes[1].scatter(ok["latency_ms_1k"], ok["cv_rmsle"], s=45, color="#2b6cb0")
    for _, r in ok.iterrows():
        axes[1].annotate(r["model"], (r["latency_ms_1k"], r["cv_rmsle"]),
                         textcoords="offset points", xytext=(5, 4), fontsize=7)
    axes[1].set_xlabel("inference latency (ms / 1k rows)")
    axes[1].set_ylabel("CV RMSLE")
    axes[1].set_title("Accuracy vs inference cost")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Model / algorithm breadth (time_v2, same folds)")
    fig.tight_layout()
    fig.savefig(FIG / "ext_model_zoo.png", dpi=200)
    plt.close(fig)
    print(ok.to_string(index=False))
    return df


if __name__ == "__main__":
    run(load_config("configs/base.yaml"))
