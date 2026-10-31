"""P1 exploratory data analysis for Bike Sharing Demand.

Usage
-----
    python3 scripts/eda.py --config configs/base.yaml

Writes at least four figures to ``reports/figures/p1_*.png`` and a machine
readable statistics file to ``reports/p1_eda_stats.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.config import load_config, path_of  # noqa: E402

WEATHER_LABELS = {
    1: "1 Clear",
    2: "2 Mist / cloudy",
    3: "3 Light precip.",
    4: "4 Heavy precip.",
}
WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PALETTE = {
    "blue": "#3B6EA5",
    "orange": "#D9822B",
    "green": "#4C9A6A",
    "red": "#C44E52",
    "grey": "#7F7F7F",
    "light_blue": "#9CC3E5",
}


def _add_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["datetime"])
    out["hour"] = dt.dt.hour.astype("int8")
    out["weekday"] = dt.dt.weekday.astype("int8")
    out["is_weekend"] = (out["weekday"] >= 5).astype("int8")
    out["month"] = dt.dt.month.astype("int8")
    out["year"] = dt.dt.year.astype("int16")
    return out


def _describe(series: pd.Series) -> dict:
    s = series.astype("float64")
    return {
        "count": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "q01": float(s.quantile(0.01)),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
        "q99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "skew": float(s.skew()),
    }


def _coverage(df: pd.DataFrame) -> dict:
    days = df["datetime"].dt.date.nunique()
    per_day = df.groupby(df["datetime"].dt.date).size()
    return {
        "unique_days": int(days),
        "expected_rows_if_24h_complete": int(days * 24),
        "missing_hours_vs_complete": int(days * 24 - len(df)),
        "days_with_24h": int((per_day == 24).sum()),
        "days_with_23h": int((per_day == 23).sum()),
        "days_with_22h": int((per_day == 22).sum()),
        "days_with_other": int((~per_day.isin([22, 23, 24])).sum()),
    }


def _corr_dict(df: pd.DataFrame, cols: list[str], method: str) -> dict[str, float]:
    corr = df[cols + ["count"]].corr(method=method)["count"].drop("count")
    return {k: round(float(v), 4) for k, v in corr.items()}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[eda] saved {path.relative_to(ROOT)}")


def plot_target_distribution(train: pd.DataFrame, fig_dir: Path) -> None:
    count = train["count"].astype("float64")
    log_count = np.log1p(count)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.hist(count, bins=60, color=PALETTE["blue"], edgecolor="white", linewidth=0.4)
    ax.axvline(count.mean(), color=PALETTE["red"], ls="--", lw=1.4, label=f"mean = {count.mean():.1f}")
    ax.axvline(count.median(), color=PALETTE["orange"], ls=":", lw=1.8, label=f"median = {count.median():.0f}")
    ax.set_title(f"Raw count (skew = {count.skew():.2f})")
    ax.set_xlabel("count")
    ax.set_ylabel("frequency")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.hist(log_count, bins=60, color=PALETTE["green"], edgecolor="white", linewidth=0.4)
    ax.axvline(log_count.mean(), color=PALETTE["red"], ls="--", lw=1.4, label=f"mean = {log_count.mean():.2f}")
    ax.axvline(log_count.median(), color=PALETTE["orange"], ls=":", lw=1.8, label=f"median = {log_count.median():.2f}")
    ax.set_title(f"log1p(count) (skew = {log_count.skew():.2f})")
    ax.set_xlabel("log1p(count)")
    ax.set_ylabel("frequency")
    ax.legend(frameon=False)

    fig.suptitle("Bike Sharing Demand — target distribution (train, n=10,886)", fontsize=12)
    fig.tight_layout()
    _save(fig, fig_dir / "p1_target_distribution.png")


def plot_hourly_profile(train: pd.DataFrame, fig_dir: Path) -> dict:
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    profiles = {}
    y_max = 0.0
    specs = [
        (1, "Working day", PALETTE["blue"]),
        (0, "Weekend / holiday", PALETTE["orange"]),
    ]
    for flag, label, color in specs:
        sub = train[train["workingday"] == flag]
        grp = sub.groupby("hour")["count"].agg(["mean", "std", "count"]).reindex(hours)
        mean = grp["mean"].to_numpy()
        sem = (grp["std"] / np.sqrt(grp["count"])).fillna(0).to_numpy()
        y_max = max(y_max, float(np.nanmax(mean)))
        ax.plot(hours, mean, color=color, lw=2.0, marker="o", ms=3.5, label=label)
        ax.fill_between(hours, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0)
        peak = int(np.nanargmax(mean))
        profiles[label] = {
            "peak_hour": peak,
            "peak_mean": float(mean[peak]),
            "min_hour": int(np.nanargmin(mean)),
            "min_mean": float(np.nanmin(mean)),
            "overall_mean": float(sub["count"].mean()),
            "rows": int(len(sub)),
        }
        ax.annotate(
            f"peak {peak}:00 = {mean[peak]:.0f}",
            xy=(peak, mean[peak]),
            xytext=(-58, 12) if flag == 1 else (6, 10),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    for start in (7, 17):
        ax.axvspan(start, start + 2, color=PALETTE["grey"], alpha=0.07, linewidth=0)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.4, 23.4)
    ax.set_ylim(0, y_max * 1.18)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean count")
    ax.set_title("Average hourly rentals: working day vs weekend/holiday (shaded = ±1 SE)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, fig_dir / "p1_hourly_profile.png")
    return profiles


def plot_weekday_hour_heatmap(train: pd.DataFrame, fig_dir: Path) -> dict:
    pivot = train.pivot_table(index="weekday", columns="hour", values="count", aggfunc="mean")
    pivot = pivot.reindex(index=range(7), columns=range(24))
    values = pivot.to_numpy()
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticks(range(7))
    ax.set_yticklabels(WEEKDAY_LABELS)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Weekday")
    ax.set_title("Mean count by weekday × hour (train)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("mean count")

    vmax = np.nanmax(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.2, color="white" if v > 0.62 * vmax else "#333333",
                )
    fig.tight_layout()
    _save(fig, fig_dir / "p1_weekday_hour_heatmap.png")
    flat = int(np.nanargmax(values))
    i, j = np.unravel_index(flat, values.shape)
    flat_min = int(np.nanargmin(values))
    i2, j2 = np.unravel_index(flat_min, values.shape)
    return {
        "max_weekday": WEEKDAY_LABELS[i],
        "max_hour": int(j),
        "max_mean": float(values[i, j]),
        "min_weekday": WEEKDAY_LABELS[i2],
        "min_hour": int(j2),
        "min_mean": float(values[i2, j2]),
        "weekday_mean": {WEEKDAY_LABELS[k]: round(float(v), 3) for k, v in pivot.mean(axis=1).items()},
    }


def _binned_mean(df: pd.DataFrame, x: str, y: str = "count", q: int = 20) -> pd.DataFrame:
    bins = pd.qcut(df[x], q=q, duplicates="drop")
    out = df.groupby(bins, observed=True).agg(x=(x, "mean"), y=(y, "mean"), n=(y, "size"))
    return out.reset_index(drop=True)


def plot_weather_temperature(train: pd.DataFrame, fig_dir: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))

    ax = axes[0, 0]
    grp = train.groupby("weather")["count"].agg(["mean", "std", "count"])
    x = np.arange(len(grp))
    sem = (grp["std"] / np.sqrt(grp["count"])).fillna(0)
    colors = [PALETTE["green"], PALETTE["blue"], PALETTE["orange"], PALETTE["red"]][: len(grp)]
    bars = ax.bar(x, grp["mean"], yerr=sem, color=colors, capsize=3, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([WEATHER_LABELS.get(int(i), str(i)) for i in grp.index], fontsize=8)
    ax.set_ylabel("Mean count")
    ax.set_title("Weather category vs mean count (error bar = ±1 SE)")
    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 6,
            f"n={int(row['count'])}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylim(0, grp["mean"].max() * 1.22)

    ax = axes[0, 1]
    hb = ax.hexbin(train["temp"], train["count"], gridsize=35, cmap="Blues", mincnt=1, bins="log")
    b = _binned_mean(train, "temp", q=18)
    ax.plot(b["x"], b["y"], color=PALETTE["red"], marker="o", ms=3.5, lw=1.5, label="binned mean")
    ax.set_xlabel("temp (normalized)")
    ax.set_ylabel("count")
    ax.set_title(f"Temperature vs count (Pearson r={train['temp'].corr(train['count']):.3f})")
    ax.legend(frameon=False, loc="upper left")
    fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.015, label="log10(rows)")

    ax = axes[1, 0]
    ax.hexbin(train["humidity"], train["count"], gridsize=35, cmap="Greens", mincnt=1, bins="log")
    b = _binned_mean(train, "humidity", q=18)
    ax.plot(b["x"], b["y"], color=PALETTE["red"], marker="o", ms=3.5, lw=1.5, label="binned mean")
    ax.set_xlabel("humidity (normalized)")
    ax.set_ylabel("count")
    ax.set_title(f"Humidity vs count (Pearson r={train['humidity'].corr(train['count']):.3f})")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 1]
    ax.hexbin(train["windspeed"], train["count"], gridsize=35, cmap="Oranges", mincnt=1, bins="log")
    b = _binned_mean(train, "windspeed", q=15)
    ax.plot(b["x"], b["y"], color=PALETTE["blue"], marker="o", ms=3.5, lw=1.5, label="binned mean")
    ax.set_xlabel("windspeed (normalized)")
    ax.set_ylabel("count")
    ax.set_title(f"Windspeed vs count (Pearson r={train['windspeed'].corr(train['count']):.3f})")
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Weather, temperature, humidity and wind vs rentals (train)", fontsize=12)
    fig.tight_layout()
    _save(fig, fig_dir / "p1_weather_temp_vs_count.png")

    return {
        "weather": {
            int(k): {
                "rows": int(v["count"]),
                "share_pct": round(100.0 * v["count"] / len(train), 2),
                "mean_count": round(float(v["mean"]), 3),
                "median_count": float(train.loc[train["weather"] == k, "count"].median()),
            }
            for k, v in grp.iterrows()
        },
        "corr_pearson": {
            "temp": round(float(train["temp"].corr(train["count"])), 4),
            "humidity": round(float(train["humidity"].corr(train["count"])), 4),
            "windspeed": round(float(train["windspeed"].corr(train["count"])), 4),
        },
    }


def plot_daily_trend(hour_ref: pd.DataFrame, train: pd.DataFrame, fig_dir: Path) -> dict:
    daily = hour_ref.groupby("dteday")["cnt"].sum().sort_index()
    is_train = daily.index.day <= 19
    train_days, test_days = daily[is_train], daily[~is_train]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.0), sharex=False)
    ax = axes[0]
    ax.plot(daily.index, daily.values, color="#C9C9C9", lw=0.8, zorder=1)
    ax.scatter(train_days.index, train_days.values, s=9, color=PALETTE["blue"], label="Kaggle train (day 1–19)", zorder=2)
    ax.scatter(test_days.index, test_days.values, s=9, color=PALETTE["orange"], label="Kaggle test (day 20–end)", zorder=2)
    ax.set_ylabel("Daily total rentals")
    ax.set_title("Daily total rentals and the Kaggle time split (UCI full data, reference only)")
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = axes[1]
    monthly = hour_ref.assign(period=hour_ref["dteday"].dt.to_period("M")).groupby("period")["cnt"].mean()
    colors = [PALETTE["blue"] if p.year == 2011 else PALETTE["orange"] for p in monthly.index]
    ax.bar([str(p) for p in monthly.index], monthly.values, color=colors, width=0.78)
    ax.set_ylabel("Mean hourly rentals")
    ax.set_title("Monthly mean hourly rentals (UCI full data): strong seasonality + 2012 growth")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.axvline(11.5, color=PALETTE["grey"], ls="--", lw=1.0)
    ax.text(0.2, monthly.max() * 0.93, "2011", color=PALETTE["blue"], fontsize=10)
    ax.text(12.2, monthly.max() * 0.93, "2012", color=PALETTE["orange"], fontsize=10)

    fig.tight_layout()
    _save(fig, fig_dir / "p1_daily_trend_split.png")
    return {
        "daily_total_mean": round(float(daily.mean()), 3),
        "daily_total_min": int(daily.min()),
        "daily_total_max": int(daily.max()),
        "monthly_mean_min": round(float(monthly.min()), 3),
        "monthly_mean_max": round(float(monthly.max()), 3),
    }


def plot_correlations(train: pd.DataFrame, fig_dir: Path) -> dict:
    cols = [
        "registered", "casual", "temp", "atemp", "hour", "season",
        "windspeed", "workingday", "holiday", "weather", "humidity",
    ]
    pear = train[cols + ["count"]].corr(method="pearson")["count"].drop("count")
    spear = train[cols + ["count"]].corr(method="spearman")["count"].drop("count")
    order = pear.sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    y = np.arange(len(order))
    ax.barh(y - 0.19, pear[order], height=0.36, color=PALETTE["blue"], label="Pearson")
    ax.barh(y + 0.19, spear[order], height=0.36, color=PALETTE["orange"], label="Spearman")
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    for label in ax.get_yticklabels():
        if label.get_text() in {"casual", "registered"}:
            label.set_color(PALETTE["red"])
            label.set_fontweight("bold")
    ax.set_xlabel("Correlation with count")
    ax.set_title(
        "Raw-feature correlation with count\n"
        "(red = leakage variables; casual + registered == count in 100% of rows)"
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, fig_dir / "p1_correlation.png")
    return {
        "pearson": {k: round(float(v), 4) for k, v in pear.items()},
        "spearman": {k: round(float(v), 4) for k, v in spear.items()},
    }


def _fold_summary(payload: dict, train: pd.DataFrame) -> list[dict]:
    dt = train["datetime"].reset_index(drop=True)
    out = []
    for f in payload["folds"]:
        tr_idx = np.asarray(f["train_idx"], dtype=int)
        va_idx = np.asarray(f["valid_idx"], dtype=int)
        out.append(
            {
                "fold": int(f["fold"]),
                "train_rows": int(len(tr_idx)),
                "valid_rows": int(len(va_idx)),
                "train_start": dt.iloc[tr_idx].min().isoformat(),
                "train_end": dt.iloc[tr_idx].max().isoformat(),
                "valid_start": dt.iloc[va_idx].min().isoformat(),
                "valid_end": dt.iloc[va_idx].max().isoformat(),
                "train_months": f.get("train_months", []),
                "valid_months": f.get("valid_months", []),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 EDA for Bike Sharing Demand")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = path_of(cfg, "processed_dir")
    raw = path_of(cfg, "raw_dir")
    fig_dir = path_of(cfg, "figures")
    reports = path_of(cfg, "reports")

    train_path = processed / "train.parquet"
    test_path = processed / "test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            "[eda] processed parquet not found; run "
            "`python3 -m src.data.make_dataset --config configs/base.yaml` first"
        )

    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)
    train = _add_time(train_raw)
    test = _add_time(test_raw)
    hour_ref = pd.read_csv(raw / "hour.csv", parse_dates=["dteday"])
    split_path = path_of(cfg, "splits")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))

    if len(train) != 10_886:
        raise SystemExit(f"[eda] unexpected train rows: {len(train)}")
    if len(test) != 6_493:
        raise SystemExit(f"[eda] unexpected test rows: {len(test)}")

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )

    print("=" * 72)
    print("P1 EDA — Bike Sharing Demand")
    print("=" * 72)
    print(f"train rows={len(train):,}  date={train['datetime'].min()} .. {train['datetime'].max()}")
    print(f"test  rows={len(test):,}  date={test['datetime'].min()} .. {test['datetime'].max()}")
    print(f"hour reference rows={len(hour_ref):,} (UCI full data; EDA only)")
    print(f"missing train={int(train.isna().sum().sum())}, test={int(test.isna().sum().sum())}")
    print(f"duplicate datetime train={int(train['datetime'].duplicated().sum())}, test={int(test['datetime'].duplicated().sum())}")
    train_cov, test_cov = _coverage(train), _coverage(test)
    print(
        f"coverage: train days={train_cov['unique_days']}, "
        f"missing hours vs 24h grid={train_cov['missing_hours_vs_complete']}; "
        f"test days={test_cov['unique_days']}, "
        f"missing hours={test_cov['missing_hours_vs_complete']}"
    )

    count_desc = _describe(train["count"])
    log_desc = _describe(np.log1p(train["count"]))
    print(
        "count: mean={:.3f} median={:.1f} min={:.0f} max={:.0f} std={:.3f} skew={:.3f}".format(
            count_desc["mean"], count_desc["median"], count_desc["min"],
            count_desc["max"], count_desc["std"], count_desc["skew"],
        )
    )
    print(
        "log1p(count): mean={:.4f} median={:.4f} std={:.4f} skew={:.3f}".format(
            log_desc["mean"], log_desc["median"], log_desc["std"], log_desc["skew"],
        )
    )

    weather_counts = train["weather"].value_counts().sort_index()
    print("weather counts:", {int(k): int(v) for k, v in weather_counts.items()})
    print("weather share %:", {int(k): round(100 * v / len(train), 2) for k, v in weather_counts.items()})

    corr_cols = ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", "hour", "weekday", "is_weekend", "casual", "registered"]
    pearson = _corr_dict(train, corr_cols, "pearson")
    spearman = _corr_dict(train, corr_cols, "spearman")
    print("Pearson corr with count:", pearson)
    print("Spearman corr with count:", spearman)
    print(f"casual+registered==count: {bool((train['casual'] + train['registered'] == train['count']).all())}")

    profiles = plot_hourly_profile(train, fig_dir)
    heat = plot_weekday_hour_heatmap(train, fig_dir)
    weather = plot_weather_temperature(train, fig_dir)
    trend = plot_daily_trend(hour_ref, train, fig_dir)
    corr = plot_correlations(train, fig_dir)
    plot_target_distribution(train, fig_dir)

    print(
        "hourly peaks: workingday {}:00 ({:.1f}), weekend/holiday {}:00 ({:.1f})".format(
            profiles["Working day"]["peak_hour"], profiles["Working day"]["peak_mean"],
            profiles["Weekend / holiday"]["peak_hour"], profiles["Weekend / holiday"]["peak_mean"],
        )
    )
    print(
        "weekday×hour max: {} {:02d}:00 mean={:.1f}; min: {} {:02d}:00 mean={:.1f}".format(
            heat["max_weekday"], heat["max_hour"], heat["max_mean"],
            heat["min_weekday"], heat["min_hour"], heat["min_mean"],
        )
    )
    print("weather mean count:", {k: v["mean_count"] for k, v in weather["weather"].items()})
    print("monthly mean hourly range: {:.2f} .. {:.2f}".format(trend["monthly_mean_min"], trend["monthly_mean_max"]))

    stats = {
        "generated_by": "scripts/eda.py",
        "config": cfg["_path"],
        "data": {
            "train": {
                "rows": int(len(train)),
                "columns": train_raw.columns.tolist(),
                "derived_columns": ["hour", "weekday", "is_weekend", "month", "year"],
                "date_min": train["datetime"].min().isoformat(),
                "date_max": train["datetime"].max().isoformat(),
                "missing_total": int(train.isna().sum().sum()),
                "missing_by_column": {k: int(v) for k, v in train.isna().sum().items()},
                "duplicate_datetime": int(train["datetime"].duplicated().sum()),
                "coverage": train_cov,
                "count": count_desc,
                "log1p_count": log_desc,
                "weather_counts": {int(k): int(v) for k, v in weather_counts.items()},
                "weather_share_pct": {int(k): round(100 * v / len(train), 3) for k, v in weather_counts.items()},
                "season_counts": {int(k): int(v) for k, v in train["season"].value_counts().sort_index().items()},
                "workingday_counts": {int(k): int(v) for k, v in train["workingday"].value_counts().sort_index().items()},
                "holiday_counts": {int(k): int(v) for k, v in train["holiday"].value_counts().sort_index().items()},
                "casual_registered_sum_equals_count": bool((train["casual"] + train["registered"] == train["count"]).all()),
                "yearly_mean_count": {int(k): round(float(v), 4) for k, v in train.groupby("year")["count"].mean().items()},
                "monthly_mean_count": {str(k): round(float(v), 4) for k, v in train.groupby(train["datetime"].dt.to_period("M"))["count"].mean().items()},
                "corr_pearson": pearson,
                "corr_spearman": spearman,
            },
            "test": {
                "rows": int(len(test)),
                "columns": test_raw.columns.tolist(),
                "derived_columns": ["hour", "weekday", "is_weekend", "month", "year"],
                "date_min": test["datetime"].min().isoformat(),
                "date_max": test["datetime"].max().isoformat(),
                "missing_total": int(test.isna().sum().sum()),
                "duplicate_datetime": int(test["datetime"].duplicated().sum()),
                "coverage": test_cov,
            },
            "hour_reference": {
                "rows": int(len(hour_ref)),
                "date_min": hour_ref["dteday"].min().isoformat(),
                "date_max": hour_ref["dteday"].max().isoformat(),
                "missing_total": int(hour_ref.isna().sum().sum()),
                "cnt_mean": round(float(hour_ref["cnt"].mean()), 4),
                "unique_days": int(hour_ref["dteday"].nunique()),
                "missing_hours_vs_complete": int(hour_ref["dteday"].nunique() * 24 - len(hour_ref)),
            },
        },
        "hourly_profile": profiles,
        "weekday_hour_heatmap": heat,
        "weather_analysis": weather,
        "daily_trend": trend,
        "correlation": corr,
        "folds": _fold_summary(split_payload, train),
        "figures": sorted(p.name for p in fig_dir.glob("p1_*.png")),
    }

    reports.mkdir(parents=True, exist_ok=True)
    stats_path = reports / "p1_eda_stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"[eda] wrote {stats_path.relative_to(ROOT)}")
    print("[eda] figures:", ", ".join(stats["figures"]))
    print("[eda] done")


if __name__ == "__main__":
    main()
