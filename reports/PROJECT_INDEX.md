# Bike Sharing Demand — 项目总索引

> 用途：答辩前快速定位“哪份文档讲什么、哪张图能放 PPT、哪些数据是当前版本”。
> 当前版本：`time_v2` 时间折，OOF 8,196 / 10,886（75.29%），主模型 LightGBM + Poisson。

## 1. 项目一句话

Kaggle **Bike Sharing Demand**：用日历、天气、节假日特征预测每小时租车量 `count`；按时间切分（每月 1–19 日训练、20 日–月末测试），主指标 **RMSLE**。

## 2. 核心结果

| 项 | 值 |
|---|---|
| 训练 / 测试 | 10,886 / 6,493 行 |
| 时间折 | `splits/time_folds.json`（time_v2，5 折 expanding-window） |
| OOF 覆盖 | 8,196 行（75.29%） |
| 主模型 | LightGBM + Poisson，800 棵，31 叶 |
| OOF RMSLE | **0.39973** |
| MAE / RMSE / R² | 46.586 / 73.611 / 0.8519 |
| 3 seeds | 0.39973 / 0.40177 / 0.40262；mean 0.40137，std 0.00148 |
| Bootstrap 95% CI | [0.3910, 0.4087] |
| 提交 | 6,493 行，无 NaN、无负值 |
| GPU 结论 | 10k 行上 GPU 更慢；200k 行约 1.6–2.2× 加速 |
| 模型族宽度 | XGBoost CUDA depth=4 0.39444，略优；CatBoost 默认弱 |

## 3. 文档地图

| 文档 | 内容 | 适合放 PPT |
|---|---|---|
| `reports/p1_data_report.md` | 数据来源、质量、EDA、time_v2 切分、特征工程、泄漏、基线 | 第 3–6 页 |
| `reports/p2_model_report.md` | 模型对比、训练协议、调参、提交、局限 | 第 7–8 页 |
| `reports/p3_eval_report.md` | RMSLE/MAE/RMSE/R²、分折、残差、分群、SHAP、badcase | 第 11 页 |
| `reports/extension_report.md` | 超参/收敛、特征消融、效率、时间漂移、多 seed、GPU、模型族宽度 | 第 9–10 页 |
| `reports/model_algorithm_comparison.md` | **模型/算法宽度对比表（GPU 素材）** | 第 9 页 |
| `reports/reference_review.md` | 参考 PPT 复盘 + 项目缺口 + 拓展建议 | 答辩思路 |
| `docs/reproduce.md` | 本地 RTX 5060 复现记录 | 答辩“可复现”证据 |
| `docs/feature_dictionary.md` | 逐特征字典 | 附录 |
| `README.md` | 项目入口、命令 | 附录 |
| `PROJECT_SPEC.md` | 共享契约（数据/切分/接口） | 附录 |

## 4. 图目录

| 图 | 用途 |
|---|---|
| `figures/p1_target_distribution.png` | 目标分布 |
| `figures/p1_hourly_profile.png` | 小时双峰 |
| `figures/p1_weekday_hour_heatmap.png` | 工作日×小时 |
| `figures/p1_weather_temp_vs_count.png` | 天气/温度 |
| `figures/p1_daily_trend_split.png` | 时间趋势与切分 |
| `figures/p1_correlation.png` | 相关性/泄漏 |
| `figures/p2_model_comparison.png` | 模型对比 |
| `figures/p2_fold_rmsle.png` | 分折 RMSLE |
| `figures/p2_feature_importance.png` | 特征重要性 |
| `figures/p3_actual_vs_pred_over_time.png` | 预测 vs 实际 |
| `figures/p3_residual_distribution.png` | 残差分布 |
| `figures/p3_residual_by_hour.png` | 分小时残差 |
| `figures/p3_residual_by_month.png` | 分月残差 |
| `figures/p3_residual_by_weather.png` | 分天气残差 |
| `figures/p3_shap_importance.png` | SHAP 重要性 |
| `figures/p3_shap_dependence.png` | SHAP 依赖 |
| `figures/p3_badcase.png` | badcase |
| `figures/ext_hyperparam.png` | 超参敏感性 |
| `figures/ext_convergence.png` | 收敛曲线 |
| `figures/ext_feature_ablation.png` | 特征消融 |
| `figures/ext_efficiency.png` | 效率 Pareto |
| `figures/ext_temporal_shift.png` | 时间漂移 |
| `figures/ext_stability.png` | 多 seed |
| `figures/ext_gpu_benchmark.png` | GPU vs CPU |
| `figures/ext_gpu_scaling.png` | 200k 行 GPU 扩展性 |
| `figures/ext_model_zoo.png` | 模型族宽度 |

## 5. 数据与产物

| 路径 | 说明 |
|---|---|
| `data/raw/train.csv` | 10,886 行，Kaggle 切分 |
| `data/raw/test.csv` | 6,493 行 |
| `data/raw/sample_submission.csv` | 提交格式 |
| `data/raw/hour.csv` | UCI 全量 17,379 行，仅 EDA/参考 |
| `data/raw/Readme.txt` | UCI 数据说明 |
| `data/processed/` | 可重建，不入 git |
| `splits/time_folds.json` | 冻结时间折，只读 |
| `reports/submission.csv` | 最终提交 |
| `reports/exp_log.csv` | 5 个 canonical 实验 |
| `reports/oof/` | 可重建，不入 git |
| `reports/eval/` | 可重建，不入 git |
| `reports/extension/` | 横向实验 CSV/JSON |
| `reports/figures/` | 所有图 |

## 6. 命令

```bash
make data      # raw -> processed + 校验 time_v2
make eda       # P1 六图
make compare   # P2 模型对比
make predict   # P2 最终提交 + OOF
make eval      # P3 指标/残差/分群
make shap      # P3 SHAP
make badcase   # P3 badcase
make report    # P3 报告（自动生成）
make ext       # 横向 A–E
make gpu       # GPU vs CPU + 200k 行扩展性
make zoo       # 模型族宽度（LightGBM/XGBoost/CatBoost/Ridge/RF）
make verify    # 55 项验收
make test      # pytest
make all       # 主流程（不含 gpu/zoo，GPU 依赖可选）
```

## 7. 本次整理/清理记录

- 旧 `time_v1` 的 50.19% OOF、旧指标、旧图已全部被 `time_v2` 覆盖。
- `reports/exp_log.csv` 已去重，只保留 5 个 canonical 实验。
- `reports/oof/` 只保留 time_v2 的 5 个模型 OOF + `final_oof.parquet`。
- 删除未使用的 `data/raw/day.csv`（UCI 日粒度数据，项目未使用）。
- 删除 `catboost_info/`、`.pytest_cache/` 等临时目录，并加入 `.gitignore`。
- 旧 airline 项目保留在 `/root/code/ml-course`，与本项目无关；本项目在 `/root/code/bike-sharing-demand`。
- 官方 Kaggle 数据仍需 `kaggle.json`；当前数据为 UCI 同源重建的精确 Kaggle 切分。

## 8. 答辩主线（12 页）

1. 封面
2. 项目结构：基础对比 + 横向延伸
3. 数据与任务
4. EDA 三条事实
5. 切分与泄漏
6. 特征工程 + 消融
7. 基础模型对比
8. 超参与收敛
9. 效率 + GPU + 模型族宽度
10. 时间漂移/持续更新
11. 评估/残差/SHAP/badcase + 多 seed
12. 结论、局限、展望

每页统一：**目的 / 设置 / 结果 / 结论 / 局限**。
