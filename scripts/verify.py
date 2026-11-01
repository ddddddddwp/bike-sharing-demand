"""Project acceptance checks for Bike Sharing Demand.

Run: python3 scripts/verify.py --config configs/base.yaml
Exit code 0 iff all checks pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config, path_of  # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def ck(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((bool(ok), name, detail))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    raw = path_of(cfg, "raw_dir")
    proc = path_of(cfg, "processed_dir")
    reports = path_of(cfg, "reports")

    tr = pd.read_csv(raw / cfg["data"]["train_file"], parse_dates=["datetime"])
    te = pd.read_csv(raw / cfg["data"]["test_file"], parse_dates=["datetime"])
    ck("train 行数 = 10,886", len(tr) == 10886, str(len(tr)))
    ck("test 行数 = 6,493", len(te) == 6493, str(len(te)))
    ck("train 含 count", "count" in tr.columns)
    ck("test 不含 count", "count" not in te.columns)
    ck("test 不含 casual/registered", not {"casual", "registered"} & set(te.columns))
    ck("train 无缺失", int(tr.isna().sum().sum()) == 0)
    ck("test 无缺失", int(te.isna().sum().sum()) == 0)

    ptr = proc / "train.parquet"
    pte = proc / "test.parquet"
    ck("processed train 存在", ptr.exists())
    ck("processed test 存在", pte.exists())

    fp = path_of(cfg, "splits")
    payload = json.loads(fp.read_text(encoding="utf-8"))
    folds = payload.get("folds", [])
    ck("时间折 = 5", len(folds) == 5, str(len(folds)))
    ck("划分版本 = time_v2", payload.get("version") == "time_v2", str(payload.get("version")))
    for i, f in enumerate(folds):
        tr_idx = np.asarray(f["train_idx"])
        va_idx = np.asarray(f["valid_idx"])
        ck(f"fold {i} 无重叠", len(np.intersect1d(tr_idx, va_idx)) == 0)
        ck(f"fold {i} 非空", len(tr_idx) > 0 and len(va_idx) > 0)
        if i:
            ck(f"fold {i} 训练集扩展", len(tr_idx) > len(folds[i - 1]["train_idx"]))

    sub = reports / "submission.csv"
    if sub.exists():
        s = pd.read_csv(sub, parse_dates=["datetime"])
        ck("submission 行数 = 6,493", len(s) == 6493, str(len(s)))
        ck("submission 列 = datetime,count", list(s.columns) == ["datetime", "count"], str(list(s.columns)))
        ck("submission 无 NaN", int(s.isna().sum().sum()) == 0)
        ck("submission count >= 0", bool((s["count"] >= 0).all()))
        ck("submission datetime 与 test 一致", bool((s["datetime"].to_numpy() == te["datetime"].to_numpy()).all()))
    else:
        ck("submission.csv 存在", False, "P2 尚未生成")

    oof = reports / "oof" / "final_oof.parquet"
    ck("final_oof.parquet 存在", oof.exists(), "" if oof.exists() else "P2 尚未生成")
    if oof.exists():
        o = pd.read_parquet(oof)
        ck("OOF 行数 >= 5,000（时间 CV 允许部分覆盖）", len(o) >= 5000, str(len(o)))
        ck("OOF 覆盖率 >= 75%", len(o) / 10886 >= 0.75, f"{len(o)/10886:.2%}")
        ck("OOF 列含 row,y,pred", {"row", "y", "pred"} <= set(o.columns), str(list(o.columns)))
        ck("OOF row 唯一", o["row"].is_unique)
        ck("OOF row 在 [0, 10886)", bool((o["row"] >= 0).all() and (o["row"] < 10886).all()))
        ck("OOF 无 NaN", int(o[["y", "pred"]].isna().sum().sum()) == 0)

    ext = reports / "extension"
    for name, min_rows in (
        ("hyperparam.csv", 10),
        ("feature_ablation.csv", 8),
        ("efficiency.csv", 6),
        ("temporal_shift.csv", 12),
        ("stability.csv", 3),
    ):
        p = ext / name
        rows = len(pd.read_csv(p)) if p.exists() else 0
        ck(f"extension/{name} 行数 >= {min_rows}", p.exists() and rows >= min_rows, str(rows))
    for name in ("ext_hyperparam.png", "ext_convergence.png", "ext_feature_ablation.png",
                 "ext_efficiency.png", "ext_temporal_shift.png", "ext_stability.png"):
        ck(f"figures/{name} 存在", (reports / "figures" / name).exists())
    ck("extension_report.md 存在", (reports / "extension_report.md").exists())

    for name in ("p1_data_report.md", "p2_model_report.md", "p3_eval_report.md",
                 "PROJECT_INDEX.md", "model_algorithm_comparison.md"):
        ck(f"{name} 存在", (reports / name).exists())
    ck("docs/reproduce.md 存在", (ROOT / "docs" / "reproduce.md").exists())

    print("=" * 64)
    passed = sum(ok for ok, _, _ in RESULTS)
    for ok, name, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  |  {detail}" if detail else ""))
    print("=" * 64)
    print(f"结论：{passed}/{len(RESULTS)} 通过")
    if passed != len(RESULTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
