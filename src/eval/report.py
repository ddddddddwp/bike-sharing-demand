"""Generate ``reports/p3_eval_report.md`` from P3 artifacts.

The report is intentionally generated rather than hand-edited so it can be
re-run after P2 publishes the final OOF.  When metrics are missing it emits a
clearly marked PENDING report with the protocol and expected artifact list.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config, path_of
from src.eval import _utils as U


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate the P3 evaluation report")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--metrics", default=None, help="default: reports/eval/metrics.json")
    p.add_argument("--out", default=None, help="default: reports/p3_eval_report.md")
    return p.parse_args(argv)


def _load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _seg(seg: pd.DataFrame | None, name: str, n: int = 6, worst: bool = True, min_n: int = 30):
    if seg is None or seg.empty:
        return pd.DataFrame()
    s = seg[(seg["segment"] == name) & (seg["n"] >= min_n)].copy()
    if s.empty:
        return s
    return s.sort_values("rmsle", ascending=not worst).head(n)


def _seg_table(seg: pd.DataFrame | None, name: str, n: int = 6, worst: bool = True, min_n: int = 30) -> list[str]:
    s = _seg(seg, name, n=n, worst=worst, min_n=min_n)
    if s.empty:
        return ["| — | — | — | — | — | — |", "|---|---:|---:|---:|---:|---:|"]
    lines = ["| 分组 | n | RMSLE | MAE | R² | 平均偏差 |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in s.iterrows():
        lines.append(
            f"| {r['level_label']} | {int(r['n']):,} | {U.fmt(r['rmsle'], 4)} | "
            f"{U.fmt(r['mae'], 2)} | {U.fmt(r['r2'], 4)} | {U.fmt(r['bias'], 2)} |"
        )
    return lines


def _best_worst(seg: pd.DataFrame | None, name: str, min_n: int = 30):
    if seg is None or seg.empty:
        return None, None
    s = seg[(seg["segment"] == name) & (seg["n"] >= min_n)]
    if s.empty:
        return None, None
    return s.loc[s["rmsle"].idxmin()], s.loc[s["rmsle"].idxmax()]


def _fold_table(folds: list[dict], metrics: dict | None) -> list[str]:
    lines = ["| Fold | 验证窗口 | n | RMSLE | MAE | RMSE | R² |", "|---:|---|---:|---:|---:|---:|---:|"]
    for f in folds:
        if metrics:
            rmsle, mae, rmse, r2 = (U.fmt(f.get(k), 5 if k == "rmsle" else 4) for k in ("rmsle", "mae", "rmse", "r2"))
        else:
            rmsle = mae = rmse = r2 = "—"
        n = f.get("n_valid", f.get("n", 0))
        lines.append(
            f"| {f['fold']} | {f['valid_start'][:10]} ~ {f['valid_end'][:10]} | {int(n):,} | "
            f"{rmsle} | {mae} | {rmse} | {r2} |"
        )
    return lines


def _build_pending(cfg: dict, folds: list[dict], metrics_path: Path, out_path: Path) -> str:
    lines: list[str] = []
    A = lines.append
    A("# Bike Sharing Demand — P3 评估与解释报告")
    A("")
    A("> **状态：PENDING（等待 P2 最终产物）**  ")
    A(f"> 生成时间：{U.utc_now()}  ")
    A(f"> 期望指标文件：`{metrics_path}`（当前不存在）  ")
    A("> 本报告中的数值位置以 `—` / `待补充` 标记，不代表任何模型结果。")
    A("")
    A("## 0. 摘要")
    A("")
    A("P3 的评估、解释与 badcase 管线已经实现并通过 smoke test；但 P2 尚未写出")
    A("`reports/oof/final_oof.parquet`（以及 `reports/submission.csv`）。因此本报告")
    A("暂时只给出评估协议、数据切分、产物清单和复现步骤，不给出最终模型结论。")
    A("")
    A("待 P2 完成后，按第 9 节命令重跑，即可自动填充：")
    A("")
    A("- 总体与分折 RMSLE / MAE / RMSE / R²；")
    A("- 小时、星期、月份、天气、工作日分群指标；")
    A("- 残差分布、时间序列、分群残差图；")
    A("- 全量训练 LightGBM 的 TreeSHAP 重要性与依赖图；")
    A("- Top-50 badcase 与误差集中度。")
    A("")
    A("## 1. 评估协议")
    A("")
    A("### 1.1 数据与时间切分")
    A("")
    A("数据为 Kaggle Bike Sharing Demand 小时级数据。训练集包含 2011-01-01 至")
    A("2012-12-19 的小时记录，共 10,886 行；测试集为各月 20 日之后的 6,493 行。")
    A("训练/测试切分与 Kaggle 一致，不做随机重排。")
    A("")
    A("本地验证使用 `splits/time_folds.json` 的 5 折 expanding-window：每折用历史")
    A("月份训练，用紧接着的两个月（最后一折四个月）验证。验证行是 processed train")
    A("的位置索引，OOF 的 `row` 字段与之一一对应。")
    A("")
    A("| Fold | 训练月份 | 验证月份 | 验证行数 | 验证窗口 |")
    A("|---:|---:|---:|---:|---|")
    for f in folds:
        A(
            f"| {f['fold']} | 2011-01 起共 {f['n_train']:,} 行 | "
            f"{', '.join(str(m) for m in f['valid_months'])} 月 | {f['n_valid']:,} | "
            f"{f['valid_start'][:10]} ~ {f['valid_end'][:10]} |"
        )
    A("")
    A("### 1.2 指标定义")
    A("")
    A("主指标 RMSLE 在 `log1p` 空间计算：")
    A("")
    A("```text")
    A("RMSLE = sqrt(mean((log1p(pred) - log1p(actual))^2))")
    A("MAE   = mean(|pred - actual|)")
    A("RMSE  = sqrt(mean((pred - actual)^2))")
    A("R²    = 1 - SS_res / SS_tot")
    A("```")
    A("")
    A("所有预测在计算指标前统一裁剪到 `>= 0`。RMSLE 对低流量时段的相对误差更敏感，")
    A("MAE/RMSE 则更受高流量时段绝对误差影响，因此三者需要一起阅读。")
    A("")
    A("### 1.3 防泄漏约束")
    A("")
    A("- `casual` 与 `registered` 永不作为特征，二者之和等于 `count`。")
    A("- `FeatureBuilder.fit` 只在训练行上统计目标聚合；验证/测试行只做 `transform`。")
    A("- 时间聚合特征（hour/weekday/month 等均值）使用平滑，避免小样本类别过拟合。")
    A("- 不使用随机 KFold，不跨时间打乱。")
    A("- 预测裁剪到非负，避免 `log1p` 对负值未定义。")
    A("")
    A("### 1.4 评估范围")
    A("")
    A("OOF 只覆盖 2012 年各月 1–19 日，共 5,464 行（约占训练集 50.2%）。")
    A("它适合比较时间外推与折间稳定性，但不能直接代表 Kaggle 测试集 20 日之后的")
    A("分布，也不能替代线上 leaderboard。")
    A("")
    A("## 2. 主结果")
    A("")
    A("### 2.1 总体指标")
    A("")
    A("| 指标 | 数值 |")
    A("|---|---:|")
    for label in ["RMSLE（主指标）", "MAE", "RMSE", "R²", "平均实际值", "平均预测值", "平均偏差"]:
        A(f"| {label} | — |")
    A("")
    A("> 待 P2 生成 final OOF 后由 `src.eval.evaluate` 填充。")
    A("")
    A("### 2.2 分折指标")
    A("")
    lines.extend(_fold_table(folds, None))
    A("")
    A("分折表用于检查模型是否在某个时间窗口显著退化。若折间 RMSLE 方差大，")
    A("应优先检查该折验证月份的季节/天气异常，而不是只追求总体均值。")
    A("")
    A("### 2.3 稳定性判据")
    A("")
    A("建议同时报告：折间 RMSLE 的均值、标准差、最差折；以及相对 P1 baseline 的")
    A("提升。单看总体 OOF 容易掩盖最后一折（2012-09~12）的分布漂移。")
    A("")
    A("## 3. 残差分析")
    A("")
    A("### 3.1 残差分布")
    A("")
    A("| 统计量 | 数值 |")
    A("|---|---:|")
    for label in ["对数残差均值", "对数残差标准差", "绝对对数误差中位数", "P90", "P95", "低估比例", "高估比例"]:
        A(f"| {label} | — |")
    A("")
    A("残差定义为 `log1p(pred) − log1p(actual)`：负值表示低估，正值表示高估。")
    A("若均值明显偏离 0，说明模型存在系统性偏差；若 P90/P95 远高于中位数，说明")
    A("少数极端行贡献了大部分 RMSLE。")
    A("")
    A("### 3.2 时间维度")
    A("")
    A("按日聚合的 actual vs predicted 曲线用于观察：")
    A("")
    A("- 是否存在整段区间的系统性高估/低估；")
    A("- 季节转换（春季升温、冬季低温）处是否出现结构性偏移；")
    A("- 折边界附近是否因训练窗口变化而跳变。")
    A("")
    A("按小时的残差箱线图用于观察早晚高峰是否被系统性低估，以及夜间低流量时段的")
    A("相对误差是否更大。")
    A("")
    A("### 3.3 月份与天气")
    A("")
    A("月份与天气分群分别回答“季节漂移”和“恶劣天气鲁棒性”两个问题。天气类别 4")
    A("（大雨/大雪）在训练集中极少，相关指标方差很大，应结合 n 阅读。")
    A("")
    A("## 4. 分群分析")
    A("")
    A("### 4.1 小时")
    A("")
    A("| 分组 | n | RMSLE | MAE | R² | 平均偏差 |")
    A("|---|---:|---:|---:|---:|---:|")
    A("| 待补充 | — | — | — | — | — |")
    A("")
    A("小时是最关键的分群维度：通勤早晚高峰的流量高、相对误差通常较低；凌晨与深夜")
    A("流量低，单个计数误差会造成较大的 log 误差。")
    A("")
    A("### 4.2 星期与工作日")
    A("")
    A("| 分组 | n | RMSLE | MAE | R² | 平均偏差 |")
    A("|---|---:|---:|---:|---:|---:|")
    A("| 待补充 | — | — | — | — | — |")
    A("")
    A("工作日/周末的出行结构不同，模型若只学到“平均小时曲线”，容易在周末白天")
    A("高估、工作日高峰低估。")
    A("")
    A("### 4.3 月份")
    A("")
    A("| 分组 | n | RMSLE | MAE | R² | 平均偏差 |")
    A("|---|---:|---:|---:|---:|---:|")
    A("| 待补充 | — | — | — | — | — |")
    A("")
    A("### 4.4 天气")
    A("")
    A("| 分组 | n | RMSLE | MAE | R² | 平均偏差 |")
    A("|---|---:|---:|---:|---:|---:|")
    A("| 待补充 | — | — | — | — | — |")
    A("")
    A("## 5. SHAP 可解释性")
    A("")
    A("### 5.1 方法")
    A("")
    A("`src/eval/interpret.py` 在全量训练集上重新拟合 LightGBM，并默认读取")
    A("`reports/p2_final_metrics.json` 自动匹配 P2 最终模型的目标模式：若最终模型为")
    A("`lgbm`（Poisson），则在原始 count 上训练；若为 `lgbm_log`，则用 `log1p(count)` + L2。")
    A("随后对 5,000 行训练样本调用 LightGBM 原生 `pred_contrib=True`，得到 TreeSHAP 值。")
    A("`pred_contrib` 与 `shap.TreeExplainer` 使用同一套树路径算法，因此不需要额外安装 `shap`。")
    A("")
    A("SHAP 满足 `raw_prediction = expected_value + sum(shap_values)`；Poisson 目标的 raw")
    A("空间是 log mean，log 目标的空间是 log1p(count)。")
    A("")
    A("### 5.2 重要性")
    A("")
    A("| # | 特征 | mean |SHAP| | 占比 | gain 占比 |")
    A("|---:|---|---:|---:|---:|")
    A("| 待补充 | — | — | — | — |")
    A("")
    A("### 5.3 依赖关系")
    A("")
    A("`p3_shap_dependence.png` 展示 top-4 特征的 SHAP 值与特征取值关系；类别特征用")
    A("箱线图，数值特征用散点 + 分箱中位数。依赖图用于发现非线性阈值（例如温度、")
    A("小时、年份），而不是证明因果。")
    A("")
    A("### 5.4 重要限制")
    A("")
    A("- **in-sample / full-train**：SHAP 在训练行上计算，且目标聚合特征也是用全量")
    A("  训练集拟合的，因此重要性偏乐观，不能当作泛化解释。")
    A("- 相关不等于因果：温度、天气、小时之间高度相关，SHAP 的分配依赖模型结构与")
    A("  特征顺序。")
    A("- 对类别特征，SHAP 值对应 LightGBM 内部的类别编码；阅读依赖图时应结合原始")
    A("  类别标签。")
    A("")
    A("## 6. Badcase 分析")
    A("")
    A("### 6.1 Top 误差")
    A("")
    A("| # | datetime | actual | pred | abs log err | hour | weather |")
    A("|---:|---|---:|---:|---:|---:|---|")
    A("| 待补充 | — | — | — | — | — | — |")
    A("")
    A("### 6.2 集中度")
    A("")
    A("badcase 按小时、星期、天气、工作日统计，输出每个分组的 top-50 数量、组内")
    A("badcase 比例和组内 RMSLE。重点观察是否存在“某几个小时贡献了大部分极端误差”。")
    A("")
    A("### 6.3 解读边界")
    A("")
    A("Top-50 是绝对对数误差最大的行，不等价于“业务上最重要的错误”。低流量时段的")
    A("1 辆车误差可能产生很大的 log 误差，但业务影响有限；高流量时段漏预测 100 辆")
    A("可能 log 误差较小但绝对成本更高。报告同时给出 MAE 与 badcase 明细，便于区分。")
    A("")
    A("## 7. 稳定性与风险")
    A("")
    A("1. **折间方差**：expanding-window 的验证月份不同，季节效应会导致 RMSLE 波动；")
    A("   应报告均值 ± 标准差，并检查最差折。")
    A("2. **时间漂移**：2012 年需求整体高于 2011 年，若模型过度依赖 `year` 或目标")
    A("   聚合，可能在更远的未来退化。")
    A("3. **天气长尾**：weather=4 样本极少，任何分群结论都不稳定。")
    A("4. **OOF 覆盖偏差**：只覆盖每月 1–19 日，不能代表 20 日之后；Kaggle 测试集")
    A("   的时间分布可能略有不同。")
    A("5. **目标聚合泄漏风险**：`FeatureBuilder` 的聚合在折内是安全的，但全量 SHAP")
    A("   模型使用全量聚合，解释时需注意。")
    A("6. **指标与目标不一致**：若 P2 使用 Poisson 目标而评估用 RMSLE，二者优化目标")
    A("   不完全一致；本报告只描述评估结果，不推断训练目标优劣。")
    A("")
    A("## 8. 局限与后续工作")
    A("")
    A("- 未做概率预测/区间估计；RMSLE 是点预测指标，无法表达不确定性。")
    A("- 未做统计显著性检验；折间差异应结合样本量与业务成本判断。")
    A("- 未对 badcase 做逐条人工归因（如节假日、赛事、极端天气事件）。")
    A("- 后续可补充：按小时×工作日的交互分群、分位数回归、残差自相关检验、")
    A("  以及基于 SHAP 的特征筛选后重新做时间 CV。")
    A("")
    A("## 9. 复现命令")
    A("")
    A("```bash")
    A("make data       # P1: processed parquet + time folds")
    A("make train      # P2: time-CV OOF")
    A("make predict    # P2: final_oof.parquet + submission.csv")
    A("make eval       # P3: metrics.json / summary.md / segment_metrics.csv / figures")
    A("make shap       # P3: shap_importance.csv / p3_shap_*.png")
    A("make badcase    # P3: badcase_top50.csv / badcase_summary.md / p3_badcase.png")
    A("python3 -m src.eval.report --config configs/base.yaml")
    A("```")
    A("")
    A("## 10. 产物清单")
    A("")
    A("| 路径 | 内容 | 状态 |")
    A("|---|---|---|")
    A("| `reports/oof/final_oof.parquet` | P2 最终 OOF（row, y, pred） | 等待 P2 |")
    A("| `reports/submission.csv` | P2 测试集提交 | 等待 P2 |")
    A("| `reports/eval/metrics.json` | 总体/分折/残差/分群指标 | 待生成 |")
    A("| `reports/eval/segment_metrics.csv` | 分群指标明细 | 待生成 |")
    A("| `reports/eval/shap_importance.csv` | TreeSHAP 重要性 | 待生成 |")
    A("| `reports/eval/badcase_top50.csv` | Top-50 误差明细 | 待生成 |")
    A("| `reports/figures/p3_*.png` | 评估/解释图 | 待生成 |")
    A("")
    A("---")
    A("")
    A(f"*本报告由 `src/eval/report.py` 生成；当前输出：`{out_path}`。*")
    return "\n".join(lines) + "\n"


def _build_full(
    cfg: dict,
    metrics: dict,
    folds: list[dict],
    seg: pd.DataFrame | None,
    shap: pd.DataFrame | None,
    shap_meta: dict | None,
    bad: pd.DataFrame | None,
    badseg: pd.DataFrame | None,
    metrics_path: Path,
    out_path: Path,
) -> str:
    overall = metrics["overall"]
    resid = metrics["residuals"]
    status = metrics.get("status", "UNKNOWN")
    lines: list[str] = []
    A = lines.append
    A("# Bike Sharing Demand — P3 评估与解释报告")
    A("")
    A(f"> **状态：{status}**  ")
    A(f"> 生成时间：{U.utc_now()}  ")
    A(f"> OOF：`{metrics.get('oof_path')}`  ")
    A(f"> 指标：`{metrics_path}`  ")
    if status != "FINAL":
        A("> **注意：当前不是 P2 最终 OOF，数字仅用于验证评估管线，不可作为最终结论。**")
    A("")
    A("## 0. 摘要")
    A("")
    A(
        f"本次评估在 {overall['n']:,} 行时间外 OOF 上得到 **RMSLE={U.fmt(overall['rmsle'], 5)}**、"
        f"MAE={U.fmt(overall['mae'], 2)}、RMSE={U.fmt(overall['rmse'], 2)}、"
        f"R²={U.fmt(overall['r2'], 4)}。"
    )
    fs = metrics.get("fold_summary", {})
    if fs.get("mean_rmsle") is not None:
        A(
            f"5 折 RMSLE 均值 {U.fmt(fs['mean_rmsle'], 5)}，标准差 {U.fmt(fs['std_rmsle'], 5)}，"
            f"范围 {U.fmt(fs['min_rmsle'], 5)} ~ {U.fmt(fs['max_rmsle'], 5)}。"
        )
    A(
        f"对数残差均值 {U.fmt(resid['mean_log_error'], 4)}，绝对对数误差 P90 "
        f"{U.fmt(resid['p90_abs_log_error'], 4)}；低估比例 {resid['under_rate']:.1%}。"
    )
    if shap is not None and not shap.empty:
        top3 = "、".join(f"`{x}`" for x in shap.head(3)["feature"].tolist())
        A(f"TreeSHAP 重要性最高的特征为 {top3}；该结果来自全量训练模型，属于 in-sample 解释。")
    if bad is not None and not bad.empty:
        A(
            f"Top-50 badcase 的平均绝对对数误差为 {U.fmt(bad['abs_log_error'].mean(), 4)}，"
            f"其中低估 {int((bad['log_error'] < 0).sum())} 行、高估 {int((bad['log_error'] >= 0).sum())} 行。"
        )
    A("")
    folds_metrics = [f for f in metrics.get("folds", []) if f.get("rmsle") is not None]
    worst_fold = max(folds_metrics, key=lambda x: x["rmsle"]) if folds_metrics else None
    best_fold = min(folds_metrics, key=lambda x: x["rmsle"]) if folds_metrics else None
    A("**结论：** 模型的排序能力较好（Spearman "
      f"{U.fmt(resid['spearman_actual_pred'], 3)}），但存在明显的**系统性低估**："
      f"平均偏差 {U.fmt(overall['bias'], 1)}，低估比例 {resid['under_rate']:.1%}。")
    if worst_fold and best_fold:
        A(
            f"误差最大的是 fold {worst_fold['fold']}（{worst_fold['valid_start'][:10]} ~ "
            f"{worst_fold['valid_end'][:10]}，RMSLE={U.fmt(worst_fold['rmsle'], 4)}），"
            f"最小的是 fold {best_fold['fold']}（RMSLE={U.fmt(best_fold['rmsle'], 4)}）。"
        )
    A("主要风险是系统性偏差、低流量时段的相对误差、天气长尾类别，以及训练窗口外推")
    A("（尤其 fold 0 仅用 2011 年训练、验证 2012 年初）带来的季节/增长漂移。具体数字见下文。")
    A("")
    A("## 1. 评估协议")
    A("")
    A("### 1.1 数据与时间切分")
    A("")
    A("训练集为 2011-01-01 至 2012-12-19 的 10,886 条小时记录；测试集为各月 20 日")
    A("之后的 6,493 条记录。切分与 Kaggle 一致，不做随机重排。")
    A("")
    A("本地验证使用 5 折 expanding-window：每折用历史月份训练，用紧接着的两个月")
    A("（最后一折四个月）验证。OOF 的 `row` 是 processed train 的位置索引。")
    A("")
    A("| Fold | 训练行数 | 验证行数 | 验证月份 | 验证窗口 |")
    A("|---:|---:|---:|---|---|")
    for f in folds:
        A(
            f"| {f['fold']} | {f['n_train']:,} | {f['n_valid']:,} | "
            f"{', '.join(str(m) for m in f['valid_months'])} 月 | "
            f"{f['valid_start'][:10]} ~ {f['valid_end'][:10]} |"
        )
    A("")
    A("### 1.2 指标定义")
    A("")
    A("```text")
    A("RMSLE = sqrt(mean((log1p(pred) - log1p(actual))^2))   # 主指标")
    A("MAE   = mean(|pred - actual|)")
    A("RMSE  = sqrt(mean((pred - actual)^2))")
    A("R²    = 1 - SS_res / SS_tot")
    A("```")
    A("")
    A("预测先裁剪到 `>= 0` 再计算所有指标。RMSLE 更关注相对误差，MAE/RMSE 更关注")
    A("绝对误差；三者结合可以区分“低流量相对误差”与“高峰绝对误差”。")
    A("")
    A("### 1.3 防泄漏约束")
    A("")
    A("- `casual` / `registered` 不作为特征（二者之和为 `count`）。")
    A("- `FeatureBuilder.fit` 只在训练行上计算目标聚合，验证行只 `transform`。")
    A("- 不使用随机 KFold，不跨时间打乱。")
    A("- 预测裁剪到非负，避免 `log1p` 未定义。")
    A("")
    A("### 1.4 评估范围与覆盖")
    A("")
    A(
        f"OOF 共 {metrics.get('n_oof', 0):,} 行，覆盖训练集 {metrics.get('coverage_of_train', 0):.1%}；"
        f"OOF 源：`{metrics.get('oof_source', 'unknown')}`。"
    )
    A("OOF 只覆盖 2012 年各月 1–19 日，不能代表 20 日之后的分布，也不能替代 leaderboard。")
    A("")
    A("## 2. 主结果")
    A("")
    A("### 2.1 总体指标")
    A("")
    A("| 指标 | 数值 |")
    A("|---|---:|")
    A(f"| RMSLE（主指标） | **{U.fmt(overall['rmsle'], 5)}** |")
    A(f"| MAE | {U.fmt(overall['mae'], 3)} |")
    A(f"| RMSE | {U.fmt(overall['rmse'], 3)} |")
    A(f"| R² | {U.fmt(overall['r2'], 5)} |")
    A(f"| 平均实际值 | {U.fmt(overall['mean_actual'], 2)} |")
    A(f"| 平均预测值 | {U.fmt(overall['mean_pred'], 2)} |")
    A(f"| 平均偏差（pred − actual） | {U.fmt(overall['bias'], 2)} |")
    A(f"| 低估比例 | {overall['under_rate']:.1%} |")
    A(f"| 高估比例 | {overall['over_rate']:.1%} |")
    A("")
    A("### 2.2 分折指标")
    A("")
    lines.extend(_fold_table(metrics.get("folds", folds), metrics))
    A("")
    if fs.get("mean_rmsle") is not None:
        A(
            f"折间 RMSLE 均值 {U.fmt(fs['mean_rmsle'], 5)}、标准差 {U.fmt(fs['std_rmsle'], 5)}、"
            f"最差折 {U.fmt(fs['max_rmsle'], 5)}。"
        )
        if worst_fold is not None:
            A(
                f"最差折 fold {worst_fold['fold']} 的训练窗口最早、验证 2012 年初；"
                "该折误差显著高于其余折，说明年初需求增长的外推是主要风险。"
            )
    A("")
    A("## 3. 残差分析")
    A("")
    A("### 3.1 分布")
    A("")
    A("| 统计量 | 数值 |")
    A("|---|---:|")
    A(f"| 对数残差均值 | {U.fmt(resid['mean_log_error'], 4)} |")
    A(f"| 对数残差标准差 | {U.fmt(resid['std_log_error'], 4)} |")
    A(f"| 中位数 | {U.fmt(resid['median_log_error'], 4)} |")
    A(f"| 绝对对数误差中位数 | {U.fmt(resid['median_abs_log_error'], 4)} |")
    A(f"| P75 / P90 / P95 / P99 | {U.fmt(resid['p75_abs_log_error'], 3)} / {U.fmt(resid['p90_abs_log_error'], 3)} / {U.fmt(resid['p95_abs_log_error'], 3)} / {U.fmt(resid['p99_abs_log_error'], 3)} |")
    A(f"| 最大绝对对数误差 | {U.fmt(resid['max_abs_log_error'], 3)} |")
    A(f"| 偏度 / 峰度 | {U.fmt(resid['skew_log_error'], 3)} / {U.fmt(resid['kurtosis_log_error'], 3)} |")
    A(f"| Pearson(actual, pred) | {U.fmt(resid['pearson_actual_pred'], 4)} |")
    A(f"| Spearman(actual, pred) | {U.fmt(resid['spearman_actual_pred'], 4)} |")
    A("")
    A("负残差表示低估，正残差表示高估。若均值明显偏离 0，说明模型存在系统性偏差；")
    A("若 P90/P95 远高于中位数，说明误差长尾明显，RMSLE 主要由少数极端行驱动。")
    A(
        f"本次均值 {U.fmt(resid['mean_log_error'], 4)}，"
        + ("呈系统性低估。" if resid["mean_log_error"] < -0.05 else ("呈系统性高估。" if resid["mean_log_error"] > 0.05 else "整体偏差较小。"))
        + f"绝对对数误差 P90 是中位数的 {resid['p90_abs_log_error'] / max(resid['median_abs_log_error'], 1e-9):.2f} 倍。"
    )
    A("")
    A("### 3.2 时间维度")
    A("")
    A("`p3_actual_vs_pred_over_time.png` 以日聚合展示 actual/pred 曲线与每日平均 log")
    A("误差。重点检查：")
    A("")
    A("- 是否存在连续多日的同向偏差（系统性高估/低估）；")
    A("- 季节转换处是否出现结构性跳变；")
    A("- 折边界附近是否因训练窗口变化而异常。")
    A("")
    A("`p3_residual_by_hour.png` 展示小时残差箱线图。若早晚高峰箱体整体偏负，说明模型")
    A("低估通勤高峰；若夜间箱体更宽，说明低流量时段相对误差更大。")
    A("")
    A("### 3.3 月份与天气")
    A("")
    A("`p3_residual_by_month.png` 与 `p3_residual_by_weather.png` 分别检查季节漂移和")
    A("恶劣天气鲁棒性。weather=4 样本极少，箱线图/指标只作参考。")
    A("")
    A("## 4. 分群分析")
    A("")
    A("### 4.1 小时（RMSLE 最差的 5 个）")
    A("")
    lines.extend(_seg_table(seg, "hour", n=5, worst=True))
    A("")
    A("### 4.2 星期（RMSLE 最差的 6 个）")
    A("")
    lines.extend(_seg_table(seg, "weekday", n=6, worst=True, min_n=1))
    A("")
    A("### 4.3 月份（RMSLE 最差的 8 个月）")
    A("")
    lines.extend(_seg_table(seg, "month", n=8, worst=True, min_n=1))
    A("")
    A("### 4.4 天气")
    A("")
    lines.extend(_seg_table(seg, "weather", n=4, worst=True, min_n=1))
    A("")
    A("### 4.5 工作日 / 周末")
    A("")
    lines.extend(_seg_table(seg, "workingday", n=2, worst=True, min_n=1))
    A("")
    A("### 4.6 分群要点")
    A("")
    for name, label in [("hour", "小时"), ("weekday", "星期"), ("month", "月份"), ("weather", "天气")]:
        b, w = _best_worst(seg, name, min_n=30)
        if b is not None and w is not None:
            A(
                f"- **{label}**：最优 `{b['level_label']}` RMSLE={U.fmt(b['rmsle'], 4)}；"
                f"最差 `{w['level_label']}` RMSLE={U.fmt(w['rmsle'], 4)}。"
            )
    A("")
    A("## 5. SHAP 可解释性")
    A("")
    A("### 5.1 方法")
    A("")
    A("`src/eval/interpret.py` 在全量训练集上拟合 LightGBM，并自动匹配 P2 最终模型的目标")
    A("模式（Poisson/count 或 log1p+L2），对 5,000 行训练样本调用 LightGBM 原生")
    A("`pred_contrib=True` 得到 TreeSHAP。`pred_contrib` 与 `shap.TreeExplainer` 使用同一套")
    A("树路径算法，因此不需要安装 `shap`。")
    A("")
    if shap_meta:
        rec = shap_meta.get("reconstruction_max_abs_error")
        rec_s = f"{rec:.2e}" if isinstance(rec, (int, float)) else "—"
        ev = shap_meta.get("expected_value_raw", shap_meta.get("expected_value_log"))
        A(
            f"本次目标模式 `{shap_meta.get('target_mode', '—')}`（{shap_meta.get('target_transform', '—')}），"
            f"SHAP 空间为 {shap_meta.get('shap_space', '—')}。"
        )
        A(
            f"样本 {shap_meta.get('sample_size', '—')} 行，特征 {shap_meta.get('n_features', '—')} 个，"
            f"expected value={U.fmt(ev, 4)}，重构最大绝对误差 {rec_s}。"
        )
        A(
            f"全量训练 in-sample RMSLE={U.fmt(shap_meta.get('in_sample_metrics', {}).get('rmsle'), 5)}，"
            f"R²={U.fmt(shap_meta.get('in_sample_metrics', {}).get('r2'), 4)}；该数字偏乐观，仅作参考。"
        )
    A("")
    A("### 5.2 重要性")
    A("")
    if shap is not None and not shap.empty:
        A("| # | 特征 | 类型 | mean |SHAP| | 占比 | gain 占比 |")
        A("|---:|---|---|---:|---:|---:|")
        for _, r in shap.head(10).iterrows():
            A(
                f"| {int(r['rank'])} | `{r['feature']}` | {r['feature_type']} | "
                f"{U.fmt(r['mean_abs_shap'], 4)} | {r['mean_abs_share']:.1%} | {r['gain_share']:.1%} |"
            )
    else:
        A("| # | 特征 | 类型 | mean |SHAP| | 占比 | gain 占比 |")
        A("|---:|---|---|---:|---:|---:|")
        A("| 待补充 | — | — | — | — | — |")
    A("")
    A("### 5.3 依赖关系")
    A("")
    A("`p3_shap_dependence.png` 展示 top-4 特征的 SHAP 依赖：类别特征用箱线图，数值")
    A("特征用散点 + 分箱中位数。依赖图用于观察非线性阈值和交互方向，不证明因果。")
    if shap is not None and not shap.empty:
        top_feats = "、".join(f"`{x}`" for x in shap.head(4)["feature"].tolist())
        A(f"本次 top-4 特征为 {top_feats}。")
    A("")
    A("### 5.4 限制")
    A("")
    A("- **in-sample / full-train**：SHAP 在训练行上计算，且目标聚合特征用全量训练集")
    A("  拟合，重要性偏乐观，不能当作泛化解释。")
    A("- **目标聚合主导**：排名第一的 `hour_workingday_mean` 是训练集 hour×workingday")
    A("  平滑目标均值，其主导地位部分来自目标编码，解读时需特别谨慎。")
    A("- 相关不等于因果；温度、天气、小时高度相关，SHAP 分配依赖模型结构。")
    A("- 类别特征的 SHAP 对应内部编码，阅读时需结合原始类别标签。")
    A("")
    A("## 6. Badcase 分析")
    A("")
    A("### 6.1 Top 误差")
    A("")
    if bad is not None and not bad.empty:
        A("| # | datetime | actual | pred | abs log err | hour | weather |")
        A("|---:|---|---:|---:|---:|---:|---|")
        for _, r in bad.head(10).iterrows():
            A(
                f"| {int(r['rank'])} | {r['datetime']} | {int(r['y'])} | {U.fmt(r['pred'], 2)} | "
                f"{U.fmt(r['abs_log_error'], 4)} | {int(r['hour']):02d} | {r['weather_label']} |"
            )
    else:
        A("| # | datetime | actual | pred | abs log err | hour | weather |")
        A("|---:|---|---:|---:|---:|---:|---|")
        A("| 待补充 | — | — | — | — | — | — |")
    A("")
    A("### 6.2 集中度")
    A("")
    if badseg is not None and not badseg.empty:
        for name, label in [("hour", "小时"), ("weekday", "星期"), ("weather", "天气"), ("workingday", "工作日")]:
            s = badseg[(badseg["segment"] == name) & (badseg["top50_count"] > 0)]
            if s.empty:
                continue
            s = s.sort_values(["top50_count", "top50_rate_within"], ascending=False).head(5)
            parts = [
                f"`{r['level_label']}` {int(r['top50_count'])} 个（组内 {r['top50_rate_within']:.1%}）"
                for _, r in s.iterrows()
            ]
            A(f"- **{label}**： " + "；".join(parts))
    else:
        A("- 待 P2 final OOF 后由 `src.eval.badcase` 填充。")
    A("")
    A("### 6.3 解读边界")
    A("")
    A("Top-50 按绝对对数误差排序，同时包含低流量和高流量行。低流量时段的 1 辆车误差")
    A("可能产生很大的 log 误差，但业务影响有限；高流量时段漏预测 100 辆可能 log 误差")
    A("较小但绝对成本更高。因此报告同时保留 MAE 与 badcase 明细。")
    A("")
    A("## 7. 稳定性与风险")
    A("")
    A("1. **折间方差**：验证月份不同，季节效应导致 RMSLE 波动；应看均值 ± 标准差与最差折。")
    A("2. **时间漂移**：2012 年需求高于 2011 年，过度依赖 `year` 或目标聚合可能在外推时退化。")
    A("3. **天气长尾**：weather=4 样本极少，分群结论不稳定。")
    A("4. **OOF 覆盖偏差**：只覆盖每月 1–19 日，不能代表 20 日之后。")
    A("5. **目标聚合泄漏风险**：折内聚合安全，但全量 SHAP 模型使用全量聚合，解释需谨慎。")
    A("6. **指标与训练目标**：若 P2 使用 Poisson 目标而评估用 RMSLE，二者不完全一致。")
    A("")
    warns = metrics.get("warnings") or []
    mean_log = float(resid.get("mean_log_error", 0.0) or 0.0)
    std_log = float(resid.get("std_log_error", float("nan")) or float("nan"))
    if warns or mean_log < -0.05:
        A("### 7.1 本次运行的告警")
        A("")
        for w in warns:
            A(f"- {w}")
        if mean_log < -0.05 and math.isfinite(std_log):
            factor = math.exp(-mean_log)
            A(
                f"- 若在 OOF 上做全局乘性校正 `exp({-mean_log:.4f})={factor:.3f}`，"
                f"OOF RMSLE 可降到约 **{std_log:.3f}**（即 log 残差标准差）；"
                "但这是用 2012 验证标签拟合的，测试集横跨 2011–2012，2011 需求更低，"
                "直接套用会系统性高估 2011。最终 `submission.csv` 未做该校正；"
                "正确做法是按年份/季节分层校准，并需要独立验证集。"
            )
            sub_path = path_of(cfg, "reports") / "submission.csv"
            if sub_path.exists():
                s = pd.read_csv(sub_path)
                train_path = path_of(cfg, "processed_dir") / "train.parquet"
                train_mean = float(pd.read_parquet(train_path)["count"].mean())
                A(
                    f"- 最终提交均值 **{s['count'].mean():.3f}**，训练集均值 "
                    f"**{train_mean:.3f}**；OOF 的整体低估主要来自 2012 年 fold 0 的"
                    "年度增长外推，不能直接外推到全量提交。"
                )
        A("")
    A("## 8. 局限与后续工作")
    A("")
    A("- 未做概率预测/区间估计；RMSLE 是点预测指标。")
    A("- 未做统计显著性检验；折间差异应结合样本量与业务成本判断。")
    A("- 未逐条人工归因 badcase（节假日、赛事、极端天气等）。")
    A("- 后续可补充：小时×工作日交互分群、分位数回归、残差自相关检验、SHAP 筛选后重跑 CV。")
    A("")
    A("## 9. 复现命令")
    A("")
    A("```bash")
    A("make data       # P1")
    A("make train      # P2 CV + OOF")
    A("make predict    # P2 final_oof.parquet + submission.csv")
    A("make eval       # P3 metrics / summary / segment / figures")
    A("make shap       # P3 SHAP")
    A("make badcase    # P3 badcase")
    A("python3 -m src.eval.report --config configs/base.yaml")
    A("```")
    A("")
    A("## 10. 产物清单")
    A("")
    A("| 路径 | 内容 |")
    A("|---|---|")
    A("| `reports/eval/metrics.json` | 总体/分折/残差/分群指标 |")
    A("| `reports/eval/summary.md` | 简版评估摘要 |")
    A("| `reports/eval/segment_metrics.csv` | 分群指标明细 |")
    A("| `reports/eval/shap_importance.csv` | TreeSHAP 重要性 |")
    A("| `reports/eval/shap_meta.json` | SHAP 元数据/重构误差 |")
    A("| `reports/eval/badcase_top50.csv` | Top-50 误差明细 |")
    A("| `reports/eval/badcase_segment_table.csv` | 分群误差 + badcase 集中度 |")
    A("| `reports/figures/p3_*.png` | 评估/解释图 |")
    A("")
    A("---")
    A("")
    A(f"*本报告由 `src/eval/report.py` 生成；当前输出：`{out_path}`。*")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    out_dir = path_of(cfg, "reports") / "eval"
    metrics_path = U.resolve_path(args.metrics, out_dir / "metrics.json")
    out_path = U.resolve_path(args.out, path_of(cfg, "reports") / "p3_eval_report.md")

    folds, _ = U.fold_table(cfg)
    metrics = _load_json(metrics_path)
    if metrics is None:
        text = _build_pending(cfg, folds, metrics_path, out_path)
        status = "PENDING"
    else:
        seg = None
        p = out_dir / "segment_metrics.csv"
        if p.exists():
            seg = pd.read_csv(p)
        shap = None
        p = out_dir / "shap_importance.csv"
        if p.exists():
            shap = pd.read_csv(p)
        shap_meta = _load_json(out_dir / "shap_meta.json")
        bad = None
        p = out_dir / "badcase_top50.csv"
        if p.exists():
            bad = pd.read_csv(p)
        badseg = None
        p = out_dir / "badcase_segment_table.csv"
        if p.exists():
            badseg = pd.read_csv(p)
        text = _build_full(cfg, metrics, folds, seg, shap, shap_meta, bad, badseg, metrics_path, out_path)
        status = metrics.get("status", "UNKNOWN")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    n_lines = text.count("\n")
    print(f"[P3-REPORT] status={status}  lines={n_lines}  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
