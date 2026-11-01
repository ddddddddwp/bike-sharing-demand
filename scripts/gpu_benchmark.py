#!/usr/bin/env python3
"""GPU vs CPU benchmark on the same Bike Sharing time folds.

This is an *optional* experiment: the main project uses LightGBM CPU because
the dataset is small. The benchmark answers "does the RTX 5060 help?" with
identical folds, features, target and hyperparameters.

Usage:
    python3 scripts/gpu_benchmark.py
    python3 scripts/gpu_benchmark.py --device cuda
"""
from __future__ import annotations

import argparse
import json
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


def run_device(cfg: dict, device: str) -> dict:
    import xgboost as xgb

    tr = pd.read_parquet(path_of(cfg, "processed_dir") / "train.parquet")
    folds, _ = load_folds(cfg)
    y = tr["count"].to_numpy(float)
    oof = np.full(len(tr), np.nan)
    secs = 0.0
    sizes, lat = [], []
    for tr_idx, va_idx in folds:
        b = FeatureBuilder().fit(tr.iloc[tr_idx])
        xt, xv = one_hot(b.transform(tr.iloc[tr_idx]), b.transform(tr.iloc[va_idx]))
        model = xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="count:poisson",
            tree_method="hist",
            device=device,
            n_jobs=4 if device == "cpu" else 1,
            random_state=42,
        )
        t0 = time.perf_counter()
        model.fit(xt, y[tr_idx], eval_set=[(xv, y[va_idx])], verbose=False)
        secs += time.perf_counter() - t0
        p = np.clip(model.predict(xv), 0.0, None)
        oof[va_idx] = p
        sizes.append(len(model.get_booster().save_raw()))
        sample = xv.head(min(1000, len(xv)))
        model.predict(sample)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            model.predict(sample)
            times.append((time.perf_counter() - t0) * 1000.0)
        lat.append(float(np.median(times)))
    mask = ~np.isnan(oof)
    return {
        "device": device,
        "cv_rmsle": rmsle(y[mask], oof[mask]),
        "cv_mae": mae(y[mask], oof[mask]),
        "cv_rmse": rmse(y[mask], oof[mask]),
        "train_seconds": secs,
        "latency_ms_1k": float(np.median(lat)),
        "model_kb": float(np.median(sizes) / 1024.0),
    }


def run_scaling(cfg: dict, n_rows: int = 200_000) -> pd.DataFrame:
    """Synthetic scaling stress test: does GPU help when rows are plentiful?

    This is NOT a model-quality experiment: rows are bootstrapped from train,
    and the split is random. It only measures training time at scale.
    """
    import xgboost as xgb

    tr = pd.read_parquet(path_of(cfg, "processed_dir") / "train.parquet")
    sample = tr.sample(n=n_rows, replace=True, random_state=42).reset_index(drop=True)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(sample))
    cut = int(len(sample) * 0.8)
    tr_idx, va_idx = idx[:cut], idx[cut:]
    b = FeatureBuilder().fit(sample.iloc[tr_idx])
    xt, xv = one_hot(b.transform(sample.iloc[tr_idx]), b.transform(sample.iloc[va_idx]))
    y = sample["count"].to_numpy(float)
    rows = []
    for dev in ("cpu", "cuda"):
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="count:poisson",
            tree_method="hist",
            device=dev,
            n_jobs=4 if dev == "cpu" else 1,
            random_state=42,
        )
        t0 = time.perf_counter()
        model.fit(xt, y[tr_idx], eval_set=[(xv, y[va_idx])], verbose=False)
        secs = time.perf_counter() - t0
        p = np.clip(model.predict(xv), 0.0, None)
        rows.append({
            "n_rows": len(sample),
            "n_features": xt.shape[1],
            "device": dev,
            "train_seconds": secs,
            "valid_rmsle": rmsle(y[va_idx], p),
        })
        print(f"[gpu-scaling] n={len(sample):,} device={dev} time={secs:.1f}s rmsle={rows[-1]['valid_rmsle']:.5f}")
    df = pd.DataFrame(rows)
    df.to_csv(EXT / "gpu_scaling.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(df["device"], df["train_seconds"], color="#2b6cb0")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, r["train_seconds"], f"{r['train_seconds']:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("training seconds")
    ax.set_title(f"Synthetic scaling stress test (n={len(sample):,}, 300 trees)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "ext_gpu_scaling.png", dpi=200)
    plt.close(fig)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="both", choices=["cpu", "cuda", "both"])
    ap.add_argument("--scaling", action="store_true", help="also run a 200k-row synthetic scaling benchmark")
    args = ap.parse_args()

    try:
        import xgboost as xgb
    except Exception as e:
        print(f"[gpu] xgboost not installed: {e}")
        print("      pip install xgboost-cu12==3.2.0")
        return

    cfg = load_config("configs/base.yaml")
    devices = ["cpu", "cuda"] if args.device == "both" else [args.device]
    rows = []
    for dev in devices:
        print(f"[gpu] running xgboost device={dev} ...")
        try:
            r = run_device(cfg, dev)
            r["status"] = "ok"
        except Exception as e:
            print(f"[gpu] {dev} FAILED: {type(e).__name__}: {e}")
            r = {"device": dev, "status": f"failed: {type(e).__name__}: {e}"}
        rows.append(r)

    # add LightGBM CPU baseline from the efficiency experiment
    eff_path = EXT / "efficiency.csv"
    if eff_path.exists():
        eff = pd.read_csv(eff_path)
        row = eff[eff["model"] == "lgbm_full"]
        if len(row):
            r = row.iloc[0]
            rows.append(
                {
                    "device": "lgbm_cpu",
                    "status": "ok",
                    "cv_rmsle": r["cv_rmsle"],
                    "cv_mae": r["cv_mae"],
                    "cv_rmse": r["cv_rmse"],
                    "train_seconds": r["train_seconds"],
                    "latency_ms_1k": r["latency_ms_1k"],
                    "model_kb": r["model_kb"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(EXT / "gpu_benchmark.csv", index=False)
    meta = {
        "xgboost_version": xgb.__version__,
        "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "note": "Same time_v2 folds, same features/target; one-hot categoricals for XGBoost.",
    }
    (EXT / "gpu_benchmark.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = df[df["status"] == "ok"].copy()
    if len(ok):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        axes[0].bar(ok["device"], ok["train_seconds"], color="#2b6cb0")
        axes[0].set_ylabel("training seconds (5 folds)")
        axes[0].set_title("Training time")
        axes[1].bar(ok["device"], ok["cv_rmsle"], color="#2b6cb0")
        axes[1].set_ylabel("CV RMSLE")
        axes[1].set_title("Accuracy")
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG / "ext_gpu_benchmark.png", dpi=200)
        plt.close(fig)
    print(df.to_string(index=False))
    if args.scaling:
        print("[gpu] scaling stress test (200k bootstrap rows, random split, 300 trees)")
        run_scaling(cfg)
    print(f"[gpu] outputs -> {EXT}")


if __name__ == "__main__":
    main()
