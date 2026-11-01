"""Shared helpers for the P3 evaluation / interpretation pipeline.

The functions here are deliberately small and dependency-light: pandas, numpy,
matplotlib and the project's frozen ``src.common`` helpers.  They do not touch
any P1/P2 owned file.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config, path_of
from src.common.metrics import mae, r2, rmse, rmsle
from src.common.split import load_folds

# ---------------------------------------------------------------------------
# Paths / IO
# ---------------------------------------------------------------------------

OOF_REQUIRED = ("row", "y", "pred")
SEGMENT_COLS = ("hour", "weekday", "month", "weather", "workingday")
EXTRA_SEGMENT_COLS = ("year_month",)
CAT_COLS = ("season", "weather", "weekday", "month")

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEATHER_LABELS = {
    1: "1 Clear",
    2: "2 Mist/Cloudy",
    3: "3 Light rain/snow",
    4: "4 Heavy rain/snow",
}


def resolve_path(p: str | Path | None, default: Path | None = None) -> Path:
    """Resolve a user-supplied path relative to the repository root."""
    if p is None:
        if default is None:
            raise ValueError("path is required")
        p = default
    out = Path(p)
    if not out.is_absolute():
        out = ROOT / out
    return out


def ensure_dir(p: str | Path) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def default_oof_path(cfg: dict) -> Path:
    return path_of(cfg, "oof_dir") / "final_oof.parquet"


def load_train(cfg: dict) -> pd.DataFrame:
    p = path_of(cfg, "processed_dir") / "train.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"processed train data not found: {p}\n"
            "Run `make data` first (P1 owns the data pipeline)."
        )
    return pd.read_parquet(p)


def load_oof(cfg: dict, path: str | Path | None = None) -> tuple[pd.DataFrame, Path, dict]:
    """Load and validate the final OOF file.

    Expected schema: ``row, y, pred``.  ``row`` is a positional index into
    ``data/processed/train.parquet``.  Duplicate rows are averaged and
    non-finite rows are dropped with diagnostics, so a slightly imperfect
    P2 artifact still produces an auditable report.
    """
    p = resolve_path(path, default_oof_path(cfg))
    if not p.exists():
        raise FileNotFoundError(
            f"final OOF not found: {p}\n"
            "P2 must run `make train` + `make predict` first.  For a smoke test "
            "only, pass `--fallback-baseline` to `src.eval.evaluate`."
        )
    df = pd.read_parquet(p)
    missing = [c for c in OOF_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"OOF {p} is missing columns {missing}; found {list(df.columns)}")

    diag: dict[str, Any] = {"path": str(p), "n_raw": int(len(df))}
    df = df.loc[:, list(OOF_REQUIRED)].copy()
    df["row"] = pd.to_numeric(df["row"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")

    nonfinite = ~np.isfinite(df[["row", "y", "pred"]]).all(axis=1)
    diag["n_nonfinite"] = int(nonfinite.sum())
    df = df.loc[~nonfinite].copy()
    if df.empty:
        raise ValueError(f"OOF {p} has no finite rows")

    df["row"] = df["row"].astype("int64")
    dup = int(df["row"].duplicated().sum())
    diag["n_duplicate_rows"] = dup
    if dup:
        df = (
            df.groupby("row", as_index=False)[["y", "pred"]]
            .mean()
            .sort_values("row")
            .reset_index(drop=True)
        )
    df = df.sort_values("row").reset_index(drop=True)
    diag["n"] = int(len(df))
    diag["row_min"] = int(df["row"].min())
    diag["row_max"] = int(df["row"].max())
    return df, p, diag


def merge_with_train(oof: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar / weather columns and target from processed train."""
    n = len(train)
    bad = (oof["row"] < 0) | (oof["row"] >= n)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} OOF row indices are outside [0, {n}); "
            "the OOF does not match data/processed/train.parquet"
        )
    meta = train.reset_index(drop=False)
    meta = meta.rename(columns={meta.columns[0]: "row"})
    meta["row"] = meta["row"].astype("int64")
    merged = oof.merge(meta, on="row", how="left", validate="one_to_one")
    if merged["datetime"].isna().any():
        raise ValueError("failed to attach train rows to OOF indices")
    merged = add_time_parts(merged)
    merged["y_train"] = merged["count"].astype("float64")
    merged["y_mismatch"] = (merged["y"] - merged["y_train"]).abs() > 1e-6
    return merged


def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["datetime"])
    out["year"] = dt.dt.year.astype("int16")
    out["month"] = dt.dt.month.astype("int8")
    out["day"] = dt.dt.day.astype("int8")
    out["hour"] = dt.dt.hour.astype("int8")
    out["weekday"] = dt.dt.weekday.astype("int8")
    out["dayofyear"] = dt.dt.dayofyear.astype("int16")
    out["is_weekend"] = (out["weekday"] >= 5).astype("int8")
    out["year_month"] = dt.dt.to_period("M").astype(str)
    return out


def fold_table(cfg: dict) -> tuple[list[dict], dict]:
    folds, payload = load_folds(cfg)
    train = load_train(cfg)
    rows: list[dict] = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        va = train.iloc[va_idx]
        meta = payload.get("folds", [])[i] if i < len(payload.get("folds", [])) else {}
        valid_months = meta.get("valid_months")
        if valid_months is None:
            valid_months = [str(x) for x in sorted(pd.to_datetime(va["datetime"]).dt.to_period("M").unique())]
        rows.append(
            {
                "fold": i,
                "train_idx": np.asarray(tr_idx, dtype=int),
                "valid_idx": np.asarray(va_idx, dtype=int),
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "valid_start": str(pd.to_datetime(va["datetime"]).min()),
                "valid_end": str(pd.to_datetime(va["datetime"]).max()),
                "valid_months": [str(x) for x in valid_months],
            }
        )
    return rows, payload


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_block(y_true, y_pred) -> dict[str, Any]:
    """Full metric block used in JSON / CSV outputs."""
    y = np.asarray(y_true, dtype="float64").reshape(-1)
    p = np.clip(np.asarray(y_pred, dtype="float64").reshape(-1), 0.0, None)
    if len(y) == 0:
        return {"n": 0}
    log_err = np.log1p(p) - np.log1p(y)
    abs_log = np.abs(log_err)
    return {
        "n": int(len(y)),
        "rmsle": float(rmsle(y, p)),
        "mae": float(mae(y, p)),
        "rmse": float(rmse(y, p)),
        "r2": float(r2(y, p)),
        "mean_actual": float(np.mean(y)),
        "mean_pred": float(np.mean(p)),
        "bias": float(np.mean(p - y)),
        "mean_log_error": float(np.mean(log_err)),
        "mae_log": float(np.mean(abs_log)),
        "p90_abs_log_error": float(np.quantile(abs_log, 0.90)),
        "under_rate": float(np.mean(log_err < 0)),
        "over_rate": float(np.mean(log_err > 0)),
    }


def residual_stats(y_true, y_pred) -> dict[str, Any]:
    y = np.asarray(y_true, dtype="float64").reshape(-1)
    p = np.clip(np.asarray(y_pred, dtype="float64").reshape(-1), 0.0, None)
    err = np.log1p(p) - np.log1p(y)
    abs_err = np.abs(err)
    return {
        "mean_log_error": float(np.mean(err)),
        "std_log_error": float(np.std(err, ddof=0)),
        "median_log_error": float(np.median(err)),
        "skew_log_error": float(pd.Series(err).skew()),
        "kurtosis_log_error": float(pd.Series(err).kurtosis()),
        "median_abs_log_error": float(np.median(abs_err)),
        "p75_abs_log_error": float(np.quantile(abs_err, 0.75)),
        "p90_abs_log_error": float(np.quantile(abs_err, 0.90)),
        "p95_abs_log_error": float(np.quantile(abs_err, 0.95)),
        "p99_abs_log_error": float(np.quantile(abs_err, 0.99)),
        "max_abs_log_error": float(np.max(abs_err)),
        "under_rate": float(np.mean(err < 0)),
        "over_rate": float(np.mean(err > 0)),
        "pearson_actual_pred": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else float("nan"),
        "spearman_actual_pred": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
    }


def level_label(segment: str, level: Any) -> str:
    try:
        v = int(level)
    except (TypeError, ValueError):
        return str(level)
    if segment == "weekday" and 0 <= v <= 6:
        return WEEKDAY_LABELS[v]
    if segment == "weather" and v in WEATHER_LABELS:
        return WEATHER_LABELS[v]
    if segment == "workingday":
        return "Working day" if v == 1 else "Weekend/Holiday"
    if segment == "month":
        return f"{v:02d}"
    if segment == "hour":
        return f"{v:02d}"
    return str(level)


def segment_table(
    df: pd.DataFrame,
    segments: Iterable[str] = SEGMENT_COLS,
    top_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute per-segment metrics, optionally enriched with top-N badcase stats."""
    total = len(df)
    rows: list[dict[str, Any]] = []
    for seg in segments:
        if seg not in df.columns:
            continue
        for level, g in df.groupby(seg, observed=True, sort=True):
            m = metric_block(g["y"], g["pred"])
            row = {
                "segment": seg,
                "level": str(level),
                "level_label": level_label(seg, level),
                "n": m["n"],
                "share": float(m["n"] / total) if total else float("nan"),
                "rmsle": m["rmsle"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "mean_actual": m["mean_actual"],
                "mean_pred": m["mean_pred"],
                "bias": m["bias"],
                "mean_log_error": m["mean_log_error"],
                "mae_log": m["mae_log"],
                "p90_abs_log_error": m["p90_abs_log_error"],
            }
            if top_mask is not None:
                mask = top_mask.reindex(g.index).fillna(False).astype(bool)
                row["top50_count"] = int(mask.sum())
                row["top50_rate_within"] = float(mask.mean()) if len(mask) else float("nan")
                row["top50_share"] = (
                    float(mask.sum() / max(int(top_mask.sum()), 1)) if top_mask.sum() else 0.0
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        # numeric levels first, then year_month strings
        out["_level_num"] = pd.to_numeric(out["level"], errors="coerce")
        out = out.sort_values(["segment", "_level_num", "level"], na_position="last")
        out = out.drop(columns="_level_num").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# JSON / formatting
# ---------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and non-finite floats."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if obj is pd.NA:
        return None
    return obj


def save_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "—"
    return f"{v:.{nd}f}"


def fmt_int(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )
    return plt


COLORS = {
    "actual": "#1f4e79",
    "pred": "#d1495b",
    "resid": "#4c72b0",
    "bad": "#c0392b",
    "neutral": "#7f8c8d",
    "accent": "#2a9d8f",
    "warn": "#e9c46a",
}


def save_fig(fig, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return p


def boxplot_by_level(ax, df: pd.DataFrame, by: str, y: str, levels=None, color=COLORS["resid"]):
    """Boxplot helper with sensible labels and zero line."""
    if levels is None:
        levels = sorted(df[by].dropna().unique())
    data = [df.loc[df[by] == lev, y].dropna().values for lev in levels]
    bp = ax.boxplot(
        data,
        positions=np.arange(len(levels)),
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1b1b1b", "linewidth": 1.2},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.8)
    ax.axhline(0, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([level_label(by, lev) for lev in levels])
    return bp
