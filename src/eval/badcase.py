"""P3 badcase analysis: largest OOF errors and error-heavy segments.

Outputs
-------
reports/eval/badcase_top50.csv
reports/eval/badcase_segment_table.csv
reports/eval/badcase_summary.md
reports/figures/p3_badcase.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config, path_of
from src.eval import _utils as U


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Badcase analysis for final OOF (P3)")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--oof", default=None, help="OOF parquet (default: reports/oof/final_oof.parquet)")
    p.add_argument("--out-dir", default=None, help="output dir (default: reports/eval)")
    p.add_argument("--fig-dir", default=None, help="figure dir (default: reports/figures)")
    p.add_argument("--top-n", type=int, default=50)
    return p.parse_args(argv)


def _top_frame(merged: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cols = [
        "row",
        "datetime",
        "y",
        "pred",
        "abs_error",
        "log_error",
        "abs_log_error",
        "hour",
        "weekday",
        "month",
        "year_month",
        "weather",
        "workingday",
        "holiday",
        "season",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
        "is_weekend",
    ]
    top = merged.sort_values(["abs_log_error", "abs_error"], ascending=False).head(top_n).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["datetime"] = pd.to_datetime(top["datetime"]).dt.strftime("%Y-%m-%d %H:%M")
    top["weekday_label"] = top["weekday"].map(lambda x: U.level_label("weekday", x))
    top["weather_label"] = top["weather"].map(lambda x: U.level_label("weather", x))
    top["workingday_label"] = top["workingday"].map(lambda x: U.level_label("workingday", x))
    keep = ["rank"] + [c for c in cols + ["weekday_label", "weather_label", "workingday_label"] if c in top.columns]
    return top.loc[:, keep].reset_index(drop=True)


def _plot_badcase(merged: pd.DataFrame, top: pd.DataFrame, seg: pd.DataFrame, out: Path, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax = axes[0, 0]
    y = np.log1p(merged["y"].to_numpy(dtype="float64"))
    p = np.log1p(merged["pred"].to_numpy(dtype="float64"))
    lim = float(max(y.max(), p.max()) * 1.03) if len(y) else 1.0
    ax.scatter(y, p, s=6, alpha=0.18, color=U.COLORS["neutral"], edgecolors="none", rasterized=True, label="all OOF")
    ty = np.log1p(top["y"].to_numpy(dtype="float64"))
    tp = np.log1p(top["pred"].to_numpy(dtype="float64"))
    ax.scatter(ty, tp, s=30, alpha=0.85, color=U.COLORS["bad"], edgecolors="white", linewidths=0.5, label=f"top {len(top)}")
    ax.plot([0, lim], [0, lim], color="#333333", lw=1.0, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("log1p(actual)")
    ax.set_ylabel("log1p(predicted)")
    ax.set_title("Actual vs predicted — badcases highlighted")
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    d = top.head(15).iloc[::-1]
    labels = [f"{dt[5:10]} {dt[11:16]}" for dt in d["datetime"]]
    ax.barh(np.arange(len(d)), d["abs_log_error"], color=U.COLORS["bad"], alpha=0.8)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("|log error|")
    ax.set_title("Top 15 absolute log errors")
    for i, v in enumerate(d["abs_log_error"]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=7)

    ax = axes[1, 0]
    hour_counts = top.groupby("hour").size().reindex(range(24), fill_value=0)
    bars = ax.bar(hour_counts.index, hour_counts.values, color=U.COLORS["bad"], alpha=0.8)
    for b, v in zip(bars, hour_counts.values):
        if v:
            ax.text(b.get_x() + b.get_width() / 2, v, str(int(v)), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour")
    ax.set_ylabel(f"count in top {len(top)}")
    ax.set_title("Badcase count by hour")

    ax = axes[1, 1]
    s = seg[(seg["segment"] == "hour") & (seg["n"] > 0)].copy()
    s["_lv"] = pd.to_numeric(s["level"], errors="coerce")
    s = s.sort_values("_lv")
    colors = [U.COLORS["bad"] if int(r["top50_count"]) > 0 else U.COLORS["neutral"] for _, r in s.iterrows()]
    ax.bar(s["_lv"], s["rmsle"], color=colors, alpha=0.8)
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour")
    ax.set_ylabel("RMSLE")
    ax.set_title("RMSLE by hour (red = contains badcases)")
    fig.suptitle("P3 badcase diagnostics — OOF", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return U.save_fig(fig, out)


def _write_summary(
    path: Path,
    merged: pd.DataFrame,
    top: pd.DataFrame,
    seg: pd.DataFrame,
    status: str,
    oof_path: Path,
    top_n: int,
):
    overall = U.metric_block(merged["y"], merged["pred"])
    lines: list[str] = []
    lines.append("# P3 Badcase 分析")
    lines.append("")
    lines.append(f"> 状态：**{status}**  ")
    lines.append(f"> OOF：`{oof_path}`  ")
    lines.append(f"> 全体 OOF：{len(merged):,} 行；RMSLE={U.fmt(overall['rmsle'], 5)}，MAE={U.fmt(overall['mae'], 2)}。  ")
    lines.append("> Badcase 定义：`|log1p(pred) − log1p(actual)|` 最大的行（默认 top 50）。")
    lines.append("")
    lines.append("## 1. 误差结构")
    lines.append("")
    lines.append("| 指标 | 全体 OOF | Top badcases |")
    lines.append("|---|---:|---:|")
    lines.append(f"| n | {len(merged):,} | {len(top):,} |")
    lines.append(f"| 平均实际值 | {U.fmt(merged['y'].mean(), 2)} | {U.fmt(top['y'].mean(), 2)} |")
    lines.append(f"| 平均预测值 | {U.fmt(merged['pred'].mean(), 2)} | {U.fmt(top['pred'].mean(), 2)} |")
    lines.append(f"| 平均绝对对数误差 | {U.fmt(np.abs(merged['log_error']).mean(), 4)} | {U.fmt(top['abs_log_error'].mean(), 4)} |")
    lines.append(f"| 低估比例 | {U.fmt((merged['log_error'] < 0).mean(), 3)} | {U.fmt((top['log_error'] < 0).mean(), 3)} |")
    lines.append("")
    lines.append("## 2. Top 10 误差")
    lines.append("")
    lines.append("| # | datetime | actual | pred | abs log err | hour | weather | workingday |")
    lines.append("|---:|---|---:|---:|---:|---:|---|---|")
    for _, r in top.head(10).iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['datetime']} | {int(r['y'])} | {U.fmt(r['pred'], 2)} | "
            f"{U.fmt(r['abs_log_error'], 4)} | {int(r['hour']):02d} | {r['weather_label']} | {r['workingday_label']} |"
        )
    lines.append("")
    lines.append("## 3. Badcase 的分群集中度")
    lines.append("")
    for seg_name, label in [("hour", "小时"), ("weekday", "星期"), ("weather", "天气"), ("workingday", "工作日")]:
        s = seg[(seg["segment"] == seg_name) & (seg["top50_count"] > 0)].copy()
        if s.empty:
            continue
        s = s.sort_values(["top50_count", "top50_rate_within"], ascending=False).head(5)
        parts = [
            f"`{r['level_label']}` {int(r['top50_count'])} 个（组内 {r['top50_rate_within']:.1%}）"
            for _, r in s.iterrows()
        ]
        lines.append(f"- **{label}**： " + "；".join(parts))
    lines.append("")
    worst = seg[(seg["segment"] == "hour") & (seg["n"] >= 30)].sort_values("rmsle", ascending=False).head(3)
    if not worst.empty:
        lines.append(
            "小时 RMSLE 最高的三个时段："
            + "；".join(
                f"`{r['level_label']}` {U.fmt(r['rmsle'], 4)}（n={int(r['n'])}）" for _, r in worst.iterrows()
            )
            + "。"
        )
    lines.append("")
    lines.append("## 4. 解读与注意")
    lines.append("")
    lines.append(
        "- Badcase 同时包含极端低流量与高峰高流量行；绝对对数误差会同时惩罚这两类，"
        "因此不能把 top 列表简单理解为“模型只在高流量时段出错”。"
    )
    lines.append(
        "- 表中只使用 OOF 预测与原始特征，不包含 `casual` / `registered`，避免把目标分解信息带入诊断结论。"
    )
    lines.append(
        "- OOF 覆盖 "
        f"{pd.to_datetime(merged['datetime']).min():%Y-%m-%d} ~ "
        f"{pd.to_datetime(merged['datetime']).max():%Y-%m-%d}，各月 "
        f"{int(pd.to_datetime(merged['datetime']).dt.day.min())}–"
        f"{int(pd.to_datetime(merged['datetime']).dt.day.max())} 日；"
        "badcase 的日期/天气集中度可能反映该切分窗口，而非全年分布。"
    )
    lines.append(
        "- 若某天气类别样本极少（如 weather=4），对应分群指标不稳定，应结合样本量阅读。"
    )
    lines.append("")
    lines.append("## 5. 产物")
    lines.append("")
    lines.append(f"- `reports/eval/badcase_top{top_n}.csv`：完整 top-{top_n} 明细")
    lines.append("- `reports/eval/badcase_segment_table.csv`：分群误差 + badcase 集中度")
    lines.append("- `reports/figures/p3_badcase.png`：散点、top 误差、小时集中度、小时 RMSLE")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    out_dir = U.ensure_dir(U.resolve_path(args.out_dir, path_of(cfg, "reports") / "eval"))
    fig_dir = U.ensure_dir(U.resolve_path(args.fig_dir, path_of(cfg, "figures")))

    train = U.load_train(cfg)
    oof, oof_path, _ = U.load_oof(cfg, args.oof)
    default_path = U.default_oof_path(cfg)
    status = "FINAL" if oof_path.resolve() == default_path.resolve() else "SMOKE"
    merged = U.merge_with_train(oof, train)
    merged["log_error"] = np.log1p(merged["pred"]) - np.log1p(merged["y"])
    merged["abs_log_error"] = merged["log_error"].abs()
    merged["abs_error"] = (merged["pred"] - merged["y"]).abs()

    top_n = int(min(max(args.top_n, 1), len(merged)))
    top = _top_frame(merged, top_n)
    top_path = out_dir / f"badcase_top{top_n}.csv"
    top.to_csv(top_path, index=False, float_format="%.6f")

    top_mask = pd.Series(False, index=merged.index)
    top_mask.loc[merged.sort_values(["abs_log_error", "abs_error"], ascending=False).head(top_n).index] = True
    seg = U.segment_table(merged, U.SEGMENT_COLS + U.EXTRA_SEGMENT_COLS, top_mask=top_mask)
    seg_path = out_dir / "badcase_segment_table.csv"
    seg.to_csv(seg_path, index=False, float_format="%.6f")

    summary_path = out_dir / "badcase_summary.md"
    _write_summary(summary_path, merged, top, seg, status, oof_path, top_n)

    plt = U.setup_matplotlib()
    fig_path = _plot_badcase(merged, top, seg, fig_dir / "p3_badcase.png", plt)

    print(f"[P3-BADCASE] status={status}  oof={oof_path}")
    print(f"[P3-BADCASE] top {top_n}: mean |log error|={top['abs_log_error'].mean():.4f}  under-rate={(top['log_error'] < 0).mean():.1%}")
    print(f"[P3-BADCASE] wrote {top_path}")
    print(f"[P3-BADCASE] wrote {seg_path}")
    print(f"[P3-BADCASE] wrote {summary_path}")
    print(f"[P3-BADCASE] wrote {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
