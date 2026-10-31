"""Regression metrics. Predictions are clipped to >= 0 before RMSLE/log metrics."""
from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64").reshape(-1)


def clip_nonneg(y_pred) -> np.ndarray:
    return np.clip(_arr(y_pred), 0.0, None)


def rmsle(y_true, y_pred) -> float:
    y = _arr(y_true)
    p = clip_nonneg(y_pred)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(clip_nonneg(y_pred) - _arr(y_true))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((clip_nonneg(y_pred) - _arr(y_true)) ** 2)))


def r2(y_true, y_pred) -> float:
    y = _arr(y_true)
    p = clip_nonneg(y_pred)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
