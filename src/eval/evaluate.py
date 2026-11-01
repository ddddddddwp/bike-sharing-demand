"""P3 evaluation: overall / fold / segment metrics and residual diagnostics.

Usage
-----
    python3 -m src.eval.evaluate --config configs/base.yaml
    python3 -m src.eval.evaluate --config configs/base.yaml --fallback-baseline

``--fallback-baseline`` is for smoke tests only: it trains a small honest
time-CV LightGBM and writes a temporary OOF inside the chosen output
directory.  It never writes ``reports/oof/final_oof.parquet``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config, path_of
from src.eval import _utils as U
from src.eval._modeling import quick_baseline_oof


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate final OOF predictions (P3)")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--oof", default=None, help="OOF parquet (default: reports/oof/final_oof.parquet)")
    p.add_argument("--out-dir", default=None, help="output dir (default: reports/eval)")
    p.add_argument("--fig-dir", default=None, help="figure dir (default: reports/figures)")
    p.add_argument(
        "--fallback-baseline",
        action="store_true",
        help="if final OOF is missing, build a temporary quick-baseline OOF (smoke only)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _load_inputs(cfg: dict, args, out_dir: Path):
    default_path = U.default_oof_path(cfg)
    oof_path = U.resolve_path(args.oof, default_path)
    diag: dict = {}
    source = "final_oof"
    if not oof_path.exists():
        if not args.fallback_baseline:
            raise SystemExit(
                f"[P3] final OOF not found: {oof_path}\n"
                "      P2 must run `make train` and `make predict` first.\n"
                "      Smoke test only: add --fallback-baseline."
            )
        print("[P3] WARNING: final OOF missing; building a temporary quick baseline (SMOKE ONLY).")
        oof, meta = quick_baseline_oof(cfg, seed=args.seed, log=True)
        oof_path = out_dir / "smoke_oof.parquet"
        oof.to_parquet(oof_path, index=False)
        diag = {"path": str(oof_path), "n": int(len(oof)), "smoke_meta": meta}
        source = "smoke_quick_baseline"
    else:
        oof, oof_path, diag = U.load_oof(cfg, oof_path)
    is_final = oof_path.resolve() == default_path.resolve()
    status = "FINAL" if is_final else "SMOKE"
    return oof, oof_path, diag, source, status


def _plot_actual_vs_time(merged: pd.DataFrame, fold_rows: list[dict], out: Path, overall: dict, plt):
    ts = pd.to_datetime(merged["datetime"])
    daily = (
        pd.DataFrame({"datetime": ts, "y": merged["y"].to_numpy(), "pred": merged["pred"].to_numpy()})
        .set_index("datetime")
        .resample("D")
        .mean()
    )
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(daily.index, daily["y"], color=U.COLORS["actual"], lw=1.6, label="Actual (daily mean)")
    ax.plot(daily.index, daily["pred"], color=U.COLORS["pred"], lw=1.4, label="Predicted (daily mean)")
    for i, f in enumerate(fold_rows):
        start = pd.Timestamp(f["valid_start"])
        end = pd.Timestamp(f["valid_end"])
        if i % 2 == 0:
            ax.axvspan(start, end, color="#dfe6e9", alpha=0.45, zorder=0)
        ax.axvline(start, color="#95a5a6", lw=0.7, ls=":")
        ax.text(start, ax.get_ylim()[1] * 0.98, f"F{f['fold']}", fontsize=8, color="#555555", va="top")
    ax.set_ylabel("Hourly rentals (daily mean)")
    ax.set_title("Actual vs predicted over time — OOF validation days")
    ax.legend(loc="upper left", ncol=2)
    ax.text(
        0.99,
        0.03,
        f"RMSLE={overall['rmsle']:.4f}   R²={overall['r2']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    daily_err = np.log1p(daily["pred"]) - np.log1p(daily["y"])
    ax2.bar(daily.index, daily_err, width=0.8, color=np.where(daily_err >= 0, U.COLORS["pred"], U.COLORS["actual"]), alpha=0.75)
    ax2.axhline(0, color="#333333", lw=0.8)
    ax2.set_ylabel("log error")
    ax2.set_xlabel("Date")
    ax2.set_title("Daily mean log error (pred − actual)", fontsize=10)
    return U.save_fig(fig, out)


def _plot_scatter(merged: pd.DataFrame, out: Path, overall: dict, plt):
    y = np.log1p(merged["y"].to_numpy(dtype="float64"))
    p = np.log1p(merged["pred"].to_numpy(dtype="float64"))
    lim = float(max(y.max(), p.max()) * 1.03) if len(y) else 1.0
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(y, p, s=6, alpha=0.28, color=U.COLORS["pred"], edgecolors="none", rasterized=True)
    ax.plot([0, lim], [0, lim], color="#333333", lw=1.0, ls="--", label="y = pred")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("log1p(actual count)")
    ax.set_ylabel("log1p(predicted count)")
    ax.set_title("Actual vs predicted (OOF, log scale)")
    ax.text(
        0.04,
        0.96,
        f"RMSLE={overall['rmsle']:.4f}\nR²={overall['r2']:.4f}\nn={overall['n']:,}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    ax.legend(loc="lower right")
    return U.save_fig(fig, out)


def _plot_residual_distribution(merged: pd.DataFrame, out: Path, resid: dict, plt):
    err = np.log1p(merged["pred"].to_numpy(dtype="float64")) - np.log1p(merged["y"].to_numpy(dtype="float64"))
    abs_err = np.abs(err)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax.hist(err, bins=80, color=U.COLORS["resid"], alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#333333", lw=1.0, ls="--")
    ax.axvline(resid["mean_log_error"], color=U.COLORS["pred"], lw=1.4, label=f"mean={resid['mean_log_error']:.3f}")
    ax.set_xlabel("log1p(pred) − log1p(actual)")
    ax.set_ylabel("Rows")
    ax.set_title("Signed log-residual distribution")
    ax.legend(loc="upper right")

    ax2.hist(abs_err, bins=80, color=U.COLORS["accent"], alpha=0.75, edgecolor="white", linewidth=0.3)
    for q, ls in [(0.50, ":"), (0.90, "--"), (0.95, "-.")]:
        v = float(np.quantile(abs_err, q))
        ax2.axvline(v, color="#333333", lw=1.0, ls=ls, label=f"p{int(q * 100)}={v:.3f}")
    ax2.set_xlabel("|log1p(pred) − log1p(actual)|")
    ax2.set_ylabel("Rows")
    ax2.set_title("Absolute log-error distribution")
    ax2.legend(loc="upper right", fontsize=8)
    return U.save_fig(fig, out)


def _plot_residual_by(merged: pd.DataFrame, by: str, out: Path, plt, title: str, levels=None):
    err_col = "_log_error"
    fig, ax = plt.subplots(figsize=(12 if by == "hour" else 9, 4.5))
    U.boxplot_by_level(ax, merged, by, err_col, levels=levels)
    if levels is None:
        levels = sorted(merged[by].dropna().unique())
    means = [merged.loc[merged[by] == lev, err_col].mean() for lev in levels]
    ax.plot(np.arange(len(levels)), means, color=U.COLORS["pred"], marker="o", ms=3.5, lw=1.0, label="mean")
    if by == "month":
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticklabels([month_names[int(x) - 1] for x in levels])
    if by == "weather":
        for i, lev in enumerate(levels):
            n = int((merged[by] == lev).sum())
            ax.text(i, ax.get_ylim()[1] * 0.98, f"n={n}", ha="center", va="top", fontsize=8, color="#555555")
    ax.set_xlabel(by)
    ax.set_ylabel("log1p(pred) − log1p(actual)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return U.save_fig(fig, out)


def _plot_hour_rmsle(seg: pd.DataFrame, out: Path, plt):
    s = seg[(seg["segment"] == "hour") & (seg["n"] > 0)].copy()
    if s.empty:
        return None
    s["_lv"] = pd.to_numeric(s["level"], errors="coerce")
    s = s.sort_values("_lv")
    fig, ax = plt.subplots(figsize=(12, 4.3))
    bars = ax.bar(s["_lv"], s["rmsle"], color=U.COLORS["accent"], alpha=0.85)
    ax.axhline(float(s["rmsle"].mean()), color=U.COLORS["pred"], lw=1.0, ls="--", label="mean across hours")
    for b, v in zip(bars, s["rmsle"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("RMSLE")
    ax.set_title("RMSLE by hour (OOF)")
    ax.legend(loc="upper left")
    return U.save_fig(fig, out)


def _pick(seg: pd.DataFrame, segment: str, min_n: int = 30, best: bool = True):
    s = seg[(seg["segment"] == segment) & (seg["n"] >= min_n)].copy()
    if s.empty:
        return None
    idx = s["rmsle"].idxmin() if best else s["rmsle"].idxmax()
    return s.loc[idx]


def _write_summary(
    path: Path,
    status: str,
    oof_path: Path,
    overall: dict,
    resid: dict,
    folds: list[dict],
    seg: pd.DataFrame,
    n_train: int,
    n_oof: int,
    scope: dict,
    warnings: list[str],
):
    lines: list[str] = []
    lines.append("# P3 评估摘要")
    lines.append("")
    if status == "FINAL":
        lines.append("> 状态：**FINAL**（使用 P2 的 `reports/oof/final_oof.parquet`）")
    else:
        lines.append("> 状态：**SMOKE**（未找到 P2 final OOF，当前为临时 quick baseline，数字不可用于最终结论）")
    lines.append(f"> OOF：`{oof_path}`  ")
    lines.append(f"> OOF 日期：{str(scope.get('date_min', '—'))[:10]} ~ {str(scope.get('date_max', '—'))[:10]}，"
                 f"各月 {scope.get('day_min', '—')}–{scope.get('day_max', '—')} 日  ")
    lines.append(f"> 训练集：{n_train:,} 行；OOF：{n_oof:,} 行（覆盖率 {n_oof / n_train:.2%}）  ")
    lines.append("> 预测统一裁剪到 `>= 0`；RMSLE 在 `log1p` 空间计算。")
    lines.append("")
    lines.append("## 1. 总体指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| RMSLE（主指标） | {U.fmt(overall['rmsle'], 5)} |")
    lines.append(f"| MAE | {U.fmt(overall['mae'], 3)} |")
    lines.append(f"| RMSE | {U.fmt(overall['rmse'], 3)} |")
    lines.append(f"| R² | {U.fmt(overall['r2'], 5)} |")
    lines.append(f"| 平均实际值 | {U.fmt(overall['mean_actual'], 2)} |")
    lines.append(f"| 平均预测值 | {U.fmt(overall['mean_pred'], 2)} |")
    lines.append(f"| 平均偏差（pred − actual） | {U.fmt(overall['bias'], 2)} |")
    lines.append("")
    lines.append("## 2. 分折指标")
    lines.append("")
    lines.append("| Fold | 验证区间 | n | RMSLE | MAE | RMSE | R² |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for f in folds:
        lines.append(
            f"| {f['fold']} | {f['valid_start'][:10]} ~ {f['valid_end'][:10]} | {f['n']:,} | "
            f"{U.fmt(f['rmsle'], 5)} | {U.fmt(f['mae'], 2)} | {U.fmt(f['rmse'], 2)} | {U.fmt(f['r2'], 4)} |"
        )
    lines.append("")
    lines.append("## 3. 残差与分群")
    lines.append("")
    lines.append(
        f"- 对数残差均值 {U.fmt(resid['mean_log_error'], 4)}，标准差 {U.fmt(resid['std_log_error'], 4)}；"
        f"绝对对数误差中位数 {U.fmt(resid['median_abs_log_error'], 4)}，P90 {U.fmt(resid['p90_abs_log_error'], 4)}。"
    )
    lines.append(
        f"- 低估比例 {resid['under_rate']:.1%}，高估比例 {resid['over_rate']:.1%}；"
        f"实际值与预测值的 Pearson 相关 {U.fmt(resid['pearson_actual_pred'], 4)}。"
    )
    for seg_name, label in [("hour", "小时"), ("weekday", "星期"), ("month", "月份"), ("weather", "天气")]:
        worst = _pick(seg, seg_name, min_n=30, best=False)
        best = _pick(seg, seg_name, min_n=30, best=True)
        if worst is not None and best is not None:
            lines.append(
                f"- **{label}**：最优 `{best['level_label']}` RMSLE={U.fmt(best['rmsle'], 4)}；"
                f"最差 `{worst['level_label']}` RMSLE={U.fmt(worst['rmsle'], 4)}。"
            )
    lines.append("")
    lines.append("## 4. 产物")
    lines.append("")
    lines.append("- `reports/eval/metrics.json`：总体、分折、残差、分群完整指标")
    lines.append("- `reports/eval/segment_metrics.csv`：hour / weekday / month / weather / workingday 分群")
    lines.append("- `reports/figures/p3_*.png`：时间序列、残差分布、分小时/月/天气残差、分群 RMSLE")
    lines.append("")
    if warnings:
        lines.append("## 5. 注意事项")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    out_dir = U.ensure_dir(U.resolve_path(args.out_dir, path_of(cfg, "reports") / "eval"))
    fig_dir = U.ensure_dir(U.resolve_path(args.fig_dir, path_of(cfg, "figures")))

    train = U.load_train(cfg)
    oof, oof_path, diag, source, status = _load_inputs(cfg, args, out_dir)
    merged = U.merge_with_train(oof, train)
    merged["_log_error"] = np.log1p(merged["pred"]) - np.log1p(merged["y"])

    warnings: list[str] = []
    if status != "FINAL":
        warnings.append("当前为 SMOKE 运行，数字仅用于验证评估管线，不代表 P2 最终模型。")
    n_mismatch = int(merged["y_mismatch"].sum())
    if n_mismatch:
        warnings.append(f"OOF 的 y 与 processed train 的 count 有 {n_mismatch} 行不一致；指标以 OOF 的 y 为准。")
    if diag.get("n_duplicate_rows"):
        warnings.append(f"OOF 存在 {diag['n_duplicate_rows']} 个重复 row，已按 row 取均值。")
    if diag.get("n_nonfinite"):
        warnings.append(f"OOF 有 {diag['n_nonfinite']} 行非有限值，已剔除。")

    overall = U.metric_block(merged["y"], merged["pred"])
    resid = U.residual_stats(merged["y"], merged["pred"])
    if overall["mean_log_error"] < -0.05:
        warnings.append(
            f"整体存在系统性低估：mean log error={overall['mean_log_error']:.4f}，"
            f"低估比例={overall['under_rate']:.1%}，平均偏差={overall['bias']:.2f}。"
        )
    elif overall["mean_log_error"] > 0.05:
        warnings.append(
            f"整体存在系统性高估：mean log error={overall['mean_log_error']:.4f}，"
            f"高估比例={overall['over_rate']:.1%}，平均偏差={overall['bias']:.2f}。"
        )

    fold_rows, fold_payload = U.fold_table(cfg)
    folds_out: list[dict] = []
    oof_rows = set(int(x) for x in merged["row"])
    for f in fold_rows:
        valid = set(int(x) for x in f["valid_idx"])
        sub = merged[merged["row"].isin(valid)]
        m = U.metric_block(sub["y"], sub["pred"]) if len(sub) else {"n": 0}
        rec = {
            "fold": f["fold"],
            "n": int(m.get("n", 0)),
            "n_expected": int(f["n_valid"]),
            "coverage": float(len(sub) / f["n_valid"]) if f["n_valid"] else float("nan"),
            "valid_start": f["valid_start"],
            "valid_end": f["valid_end"],
            "valid_months": f["valid_months"],
            **{k: m.get(k) for k in ("rmsle", "mae", "rmse", "r2", "mean_actual", "mean_pred", "bias")},
        }
        folds_out.append(rec)
        if rec["n"] == 0:
            warnings.append(f"Fold {f['fold']} 在 OOF 中没有可用行。")
        elif rec["coverage"] < 0.999:
            warnings.append(f"Fold {f['fold']} OOF 覆盖率仅 {rec['coverage']:.1%}。")

    valid_union = set()
    for f in fold_rows:
        valid_union.update(int(x) for x in f["valid_idx"])
    missing_rows = valid_union - oof_rows
    if missing_rows:
        warnings.append(f"冻结折的验证行中有 {len(missing_rows)} 行未出现在 OOF 中。")

    seg = U.segment_table(merged, U.SEGMENT_COLS + U.EXTRA_SEGMENT_COLS)
    seg_path = out_dir / "segment_metrics.csv"
    seg.to_csv(seg_path, index=False, float_format="%.6f")

    plt = U.setup_matplotlib()
    figures: list[str] = []
    figures.append(str(_plot_actual_vs_time(merged, fold_rows, fig_dir / "p3_actual_vs_pred_over_time.png", overall, plt)))
    figures.append(str(_plot_scatter(merged, fig_dir / "p3_actual_vs_pred_scatter.png", overall, plt)))
    figures.append(str(_plot_residual_distribution(merged, fig_dir / "p3_residual_distribution.png", resid, plt)))
    figures.append(str(_plot_residual_by(merged, "hour", fig_dir / "p3_residual_by_hour.png", plt, "Log residual by hour")))
    figures.append(str(_plot_residual_by(merged, "month", fig_dir / "p3_residual_by_month.png", plt, "Log residual by month")))
    figures.append(str(_plot_residual_by(merged, "weather", fig_dir / "p3_residual_by_weather.png", plt, "Log residual by weather")))
    weekday_fig = _plot_residual_by(merged, "weekday", fig_dir / "p3_residual_by_weekday.png", plt, "Log residual by weekday")
    if weekday_fig:
        figures.append(str(weekday_fig))
    hour_rmsle_fig = _plot_hour_rmsle(seg, fig_dir / "p3_segment_rmsle_hour.png", plt)
    if hour_rmsle_fig:
        figures.append(str(hour_rmsle_fig))

    fold_rmsle = [f["rmsle"] for f in folds_out if f.get("rmsle") is not None]
    oof_dt = pd.to_datetime(merged["datetime"])
    oof_scope = {
        "date_min": str(oof_dt.min()),
        "date_max": str(oof_dt.max()),
        "n_months": int(oof_dt.dt.to_period("M").nunique()),
        "day_min": int(oof_dt.dt.day.min()),
        "day_max": int(oof_dt.dt.day.max()),
    }
    payload = {
        "status": status,
        "generated_at": U.utc_now(),
        "config": cfg.get("_path"),
        "oof_path": str(oof_path),
        "oof_source": source,
        "oof_diagnostics": diag,
        "oof_scope": oof_scope,
        "n_train": int(len(train)),
        "n_oof": int(len(merged)),
        "coverage_of_train": float(len(merged) / len(train)),
        "target": "count",
        "metric_primary": "rmsle",
        "prediction_clip": "max(pred, 0)",
        "overall": overall,
        "residuals": resid,
        "folds": folds_out,
        "fold_summary": {
            "n_folds": int(len(folds_out)),
            "mean_rmsle": float(np.mean(fold_rmsle)) if fold_rmsle else None,
            "std_rmsle": float(np.std(fold_rmsle)) if fold_rmsle else None,
            "min_rmsle": float(np.min(fold_rmsle)) if fold_rmsle else None,
            "max_rmsle": float(np.max(fold_rmsle)) if fold_rmsle else None,
        },
        "segment_metrics_path": str(seg_path),
        "figures": figures,
        "warnings": warnings,
        "notes": [
            "RMSLE / MAE / RMSE / R² 均基于 OOF 验证行，预测先裁剪到 >= 0。",
            "分折指标按 splits/time_folds.json 的 valid_idx 选取 OOF 行；折间验证月份不重叠。",
            "分群指标基于 processed train 的 datetime / weather / workingday 字段。",
            f"OOF 覆盖 {oof_scope['date_min'][:10]} ~ {oof_scope['date_max'][:10]}，"
            f"各月 {oof_scope['day_min']}–{oof_scope['day_max']} 日；不能代表 20 日之后的分布。",
        ],
    }
    metrics_path = out_dir / "metrics.json"
    U.save_json(metrics_path, payload)

    summary_path = out_dir / "summary.md"
    _write_summary(
        summary_path,
        status,
        oof_path,
        overall,
        resid,
        folds_out,
        seg,
        len(train),
        len(merged),
        oof_scope,
        warnings,
    )

    print(f"[P3] status={status}  oof={oof_path}")
    print(
        f"[P3] overall n={overall['n']:,}  RMSLE={overall['rmsle']:.5f}  "
        f"MAE={overall['mae']:.3f}  RMSE={overall['rmse']:.3f}  R2={overall['r2']:.5f}"
    )
    for f in folds_out:
        print(
            f"[P3] fold {f['fold']}: n={f['n']:,}/{f['n_expected']:,}  "
            f"RMSLE={U.fmt(f.get('rmsle'), 5)}  R2={U.fmt(f.get('r2'), 4)}"
        )
    print(f"[P3] wrote {metrics_path}")
    print(f"[P3] wrote {summary_path}")
    print(f"[P3] wrote {seg_path}")
    print(f"[P3] wrote {len(figures)} figures under {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
