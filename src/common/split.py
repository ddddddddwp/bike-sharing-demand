"""Load frozen time-based folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.config import ROOT, path_of


def load_folds(cfg: dict):
    p = path_of(cfg, "splits") if "splits" in cfg.get("paths", {}) else Path(cfg["split"]["folds_file"])
    if not p.is_absolute():
        p = ROOT / p
    payload = json.loads(p.read_text(encoding="utf-8"))
    folds = [(np.asarray(f["train_idx"], dtype=int), np.asarray(f["valid_idx"], dtype=int))
             for f in payload["folds"]]
    return folds, payload
