"""Download the official Kaggle Bike Sharing Demand data.

Prerequisite: a Kaggle API token at one of:
  ~/.kaggle/kaggle.json
  ~/.config/kaggle/kaggle.json
Download it from Kaggle: Account -> Settings -> API -> Create New API Token.

Usage:
    python3 scripts/fetch_kaggle.py
    python3 scripts/fetch_kaggle.py --replace   # replace data/raw/train.csv/test.csv after checks
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = RAW / "kaggle"
COMPETITION = "bike-sharing-demand"


def _token_paths() -> list[Path]:
    return [Path.home() / ".kaggle" / "kaggle.json", Path.home() / ".config" / "kaggle" / "kaggle.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true", help="replace data/raw/train.csv/test.csv after validation")
    args = ap.parse_args()

    token = next((p for p in _token_paths() if p.exists()), None)
    if token is None:
        print("[kaggle] 找不到 kaggle.json。")
        print("  1) 登录 https://www.kaggle.com/")
        print("  2) 右上角头像 -> Settings -> API -> Create New API Token")
        print("  3) 把下载的 kaggle.json 放到 ~/.kaggle/kaggle.json")
        print("  4) 重新运行：python3 scripts/fetch_kaggle.py")
        sys.exit(2)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] 使用 token: {token}")
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(OUT)]
    print("[kaggle] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    zips = sorted(OUT.glob("*.zip"))
    if zips:
        for z in zips:
            print(f"[kaggle] unzip {z.name}")
            with zipfile.ZipFile(z) as f:
                f.extractall(OUT)

    tr = pd.read_csv(OUT / "train.csv", parse_dates=["datetime"])
    te = pd.read_csv(OUT / "test.csv", parse_dates=["datetime"])
    print(f"[kaggle] train={tr.shape} test={te.shape}")
    assert len(tr) == 10886, f"unexpected train rows: {len(tr)}"
    assert len(te) == 6493, f"unexpected test rows: {len(te)}"
    assert {"casual", "registered", "count"} <= set(tr.columns)
    assert "count" not in te.columns
    print("[kaggle] 行数/列检查通过")

    if args.replace:
        shutil.copy2(OUT / "train.csv", RAW / "train.csv")
        shutil.copy2(OUT / "test.csv", RAW / "test.csv")
        shutil.copy2(OUT / "sample_submission.csv", RAW / "sample_submission.csv")
        print("[kaggle] 已替换 data/raw/train.csv / test.csv / sample_submission.csv")
        print("[kaggle] 请重新运行：make data && make compare && make predict && make eval && make report")


if __name__ == "__main__":
    main()
