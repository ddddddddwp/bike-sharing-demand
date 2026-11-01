# Bike Sharing Demand — P2 模型与训练报告

> 角色：P2（Models & Training）  
> 项目：`/root/code/bike-sharing-demand`  
> 主指标：RMSLE；辅助指标：MAE、RMSE、R²  
> 报告日期：2026-11-01

## 1. 任务与结论

预测每小时共享单车租借量 `count`，属于计数回归 + 时间序列问题。数据按时间切分：训练集为每月 1—19 日，测试集为每月 20 日—月末，因此不能使用随机 KFold，也不能把未来月份混入训练。

P2 的目标是：在冻结的 5 折扩展窗口时间 CV 上比较 `mean`、`ridge`、`rf`、`lgbm`、`lgbm_log`；按 RMSLE 选择最佳模型；在全量 10,886 行上重新拟合并预测 6,493 行测试集；产出 OOF、实验日志、提交文件、图表和本报告。

**核心结论：**

- 最佳模型：**LightGBM + Poisson 目标（原始计数）**；
- 5 折 pooled OOF：**RMSLE = 0.39973**，MAE = 46.586，RMSE = 73.611，R² = 0.8519；
- 最佳 run_id：`lgbm_s42_20261101_094909_5d6e8f3390`；
- 提交文件 `reports/submission.csv`：6,493 行，无 NaN、无负值；
- 预测范围 2.286—942.863，均值 187.656，中位数 145.952；
- OOF 覆盖 8,196 / 10,886 行（75.29%），覆盖 2011-07 至 2012-12；前 6 个月没有更早的训练窗口，无法产生严格的时间 OOF。

## 2. 数据与时间切分

### 2.1 数据规模

| 文件 | 行数 | 说明 |
|---|---:|---|
| `data/raw/train.csv` | 10,886 | 含 `count`，含 `casual`/`registered`（不作为特征） |
| `data/raw/test.csv` | 6,493 | 不含 `count` |
| `data/raw/sample_submission.csv` | 6,493 | `datetime,count` |
| `data/processed/train.parquet` | 10,886 | 清洗后训练数据 |
| `data/processed/test.parquet` | 6,493 | 清洗后测试数据 |

训练集时间范围 2011-01-01 00:00 至 2012-12-19 23:00；测试集 2011-01-20 00:00 至 2012-12-31 23:00。

训练目标统计：均值 191.574，标准差 181.144，最小 1，中位数 145，最大 977。分年份看，2011 年均值 144.223（5,422 行），2012 年均值 238.561（5,464 行），存在明显年度增长。新 split 中 fold 1 跨越 2011-11 至 2012-02、fold 2 覆盖 2012 年春季，这两折误差较高；fold 0 只在 2011 年内做季节外推，反而较容易。

### 2.2 训练/测试的时间结构

Kaggle 切分规则是“每月前 19 天训练、20 日之后测试”。因此：

- 训练数据只包含每月 1—19 日；
- 测试数据只包含每月 20—31 日；
- 训练集和测试集在“月内日期”维度上不重叠；
- 测试集包含训练期末尾之后的未来时段（2012-12-20 至 2012-12-31）。

时间 CV 的验证月份虽然位于未来，但验证样本仍然只来自每月 1—19 日，无法直接模拟测试集的“月末”分布。该限制在第 9 节详细讨论。

### 2.3 冻结的 5 折时间 CV

`splits/time_folds.json`（split 版本 `time_v2`）给出 5 个扩展窗口折，行号是 `data/processed/train.parquet` 的位置索引：

| Fold | 训练月份 | 验证月份 | n_train | n_valid |
|---:|---|---|---:|---:|
| 0 | 2011-01 — 2011-06 | 2011-07 — 2011-10 | 2,690 | 1,820 |
| 1 | 2011-01 — 2011-10 | 2011-11 — 2012-02 | 4,510 | 1,820 |
| 2 | 2011-01 — 2012-02 | 2012-03 — 2012-06 | 6,330 | 1,821 |
| 3 | 2011-01 — 2012-06 | 2012-07 — 2012-09 | 8,151 | 1,368 |
| 4 | 2011-01 — 2012-09 | 2012-10 — 2012-12 | 9,519 | 1,367 |

5 个验证区间首尾相接，覆盖位置索引 2,690—10,885，共 8,196 行。前 2,690 行（2011-01 至 2011-06）在所有折中都只能作为训练数据，没有更早窗口可用于验证，所以 OOF 不是全量 10,886 行。这是扩展窗口时间 CV 的正常结果，不是数据缺失；`p2_final_metrics.json` 明确记录了覆盖率。

## 3. 特征工程（冻结，未修改）

P2 直接使用 P1 的 `src/features/build.py::FeatureBuilder`，没有修改特征代码。每折执行：

```python
builder = FeatureBuilder().fit(train_df.iloc[train_idx])
X_train = builder.transform(train_df.iloc[train_idx])
X_valid = builder.transform(train_df.iloc[valid_idx])
```

关键点：

- `fit` 只看到当前折训练行，目标聚合统计不会接触验证集标签；
- 输出 36 个特征，不包含 `datetime`、`count`、`casual`、`registered`；
- `casual` 与 `registered` 从未作为特征使用，避免目标泄漏；
- 类别列：`season`、`weather`、`weekday`、`month` 为 pandas `category`，`hour_x_workingday` 为字符串类别；
- 目标聚合：`hour_mean`、`weekday_mean`、`month_mean`、`season_mean`、`weather_mean`、`workingday_mean`、`hour_workingday_mean`，平滑参数 `smooth=20`，仅由训练折 `count` 计算；
- 时间特征：年、月、日、小时、星期、年内日序、年内周序、是否周末、是否高峰、是否夜间，以及小时/星期/月/日序的 sin/cos 编码；
- 交互特征：`hour_x_workingday`、`temp_x_season`、`humidity_x_weather`；
- 未加入任何目标滞后或滚动统计，模型无法直接利用前一小时/前一天租借量。

最终模型在全量训练集上重新 `FeatureBuilder.fit(all)`，再对测试集 `transform`；测试集目标聚合值只来自训练集，不使用测试标签。

## 4. 模型与超参数

五类模型使用同一套折安全特征，差异主要在目标变换和编码方式。

| 模型 | 目标变换 | 类别处理 | 关键超参数 |
|---|---|---|---|
| `mean` | 无 | 不使用特征 | 训练折全局均值 |
| `ridge` | `log1p(count)`，预测后 `expm1` | 独热编码，未知类别忽略 | `alpha=1.0`；数值特征标准化 |
| `rf` | `log1p(count)`，预测后 `expm1` | 有序编码，未知类别为 -1 | `n_estimators=300`，`min_samples_leaf=3`，`n_jobs=-1` |
| `lgbm` | 无（原始计数） | pandas `category`，未知类别为缺失 | `objective=poisson`，`n_estimators=800`，`learning_rate=0.05`，`num_leaves=31`，`min_child_samples=30`，`subsample=0.8`，`subsample_freq=1`，`colsample_bytree=0.8` |
| `lgbm_log` | `log1p(count)`，预测后 `expm1` | 同 `lgbm` | 同上，但 `objective=regression` |

实现细节：

- `mean`：每折只用训练折 `count` 的算术平均作为常数预测；
- `ridge`：`OneHotEncoder` 处理全部类别列（含 `hour_x_workingday`），`StandardScaler` 处理数值列，拟合 `log1p(count)`；
- `rf`：`OrdinalEncoder` 将类别映射为整数，随机森林拟合 `log1p(count)`，不做标准化；
- `lgbm`：LightGBM 直接拟合原始 `count`，Poisson 目标适合非负计数；类别列使用训练折类别集合，未见类别变为缺失值；
- `lgbm_log`：同一套 LightGBM 参数，但目标改为 `log1p(count)`，用于和 Poisson 对比；
- 所有模型预测后统一 `clip(pred, 0, None)`；
- 所有随机模型固定 `random_state=seed=42`；
- 未使用验证集 early stopping，避免验证标签进入模型选择；树数量固定为配置值。

## 5. 验证协议与指标

每折流程：用训练行 `fit` 全新 `FeatureBuilder` → `transform` 训练/验证行 → 拟合模型 → 预测验证行并裁剪到 `>= 0` → 写回 OOF → 计算该折指标。所有折完成后，在 OOF 覆盖的 8,196 行上计算 pooled 指标。pooled 指标比“各折指标简单平均”更能反映整体误差，因为 fold 0/1/2 各有约 1,820 行、fold 3/4 约 1,367 行，各折权重并不相等。报告同时给出两种口径。

指标定义：

- RMSLE = `sqrt(mean((log1p(pred) - log1p(actual))^2))`
- MAE = `mean(|pred - actual|)`
- RMSE = `sqrt(mean((pred - actual)^2))`
- R² = `1 - SS_res / SS_tot`

所有指标计算前都将预测裁剪到 `>= 0`。

OOF 说明：8,196 行，覆盖率 75.29%，覆盖 2011-07 至 2012-12；未覆盖 2011-01 至 2011-06（2,690 行），因为 fold 0 的最早训练窗口只有 2011 年上半年，没有更早数据可验证。`reports/oof/final_oof.parquet` 只包含这 8,196 行真实时间 CV 预测，列严格为 `row,y,pred`，没有为 2011 年上半年伪造预测。

## 6. 时间 CV 结果

### 6.1 模型总表

数据来源：`reports/p2_model_comparison.csv`，按 pooled OOF RMSLE 升序。

| Rank | Model | 目标 | CV RMSLE | CV MAE | CV RMSE | CV R² | 各折 RMSLE 平均 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **lgbm** | 原始 count / Poisson | **0.39973** | **46.586** | **73.611** | **0.8519** | 0.37519 |
| 2 | lgbm_log | log1p / regression | 0.41478 | 47.575 | 74.523 | 0.8482 | 0.38795 |
| 3 | rf | log1p / RandomForest | 0.45656 | 55.333 | 85.280 | 0.8012 | 0.43055 |
| 4 | ridge | log1p / Ridge | 0.60114 | 71.781 | 113.624 | 0.6471 | 0.53769 |
| 5 | mean | 全局均值 | 1.40600 | 149.274 | 200.994 | -0.1043 | 1.40044 |

对应 run_id（详见 `reports/p2_model_comparison.csv`）：`lgbm_s42_20261101_094909_5d6e8f3390`、`lgbm_log_s42_20261101_094915_85eeb9f451`、`rf_s42_20261101_094905_49f747a4ee`、`ridge_s42_20261101_094901_6741fcd8af`、`mean_s42_20261101_094901_ede0c8ef7c`。

### 6.2 各折 RMSLE

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| lgbm | 0.33001 | 0.56155 | 0.41470 | 0.25501 | 0.31468 |
| lgbm_log | 0.34153 | 0.60190 | 0.40285 | 0.26900 | 0.32447 |
| rf | 0.38770 | 0.62880 | 0.47474 | 0.28531 | 0.37619 |
| ridge | 0.94378 | 0.60349 | 0.41977 | 0.38046 | 0.34095 |
| mean | 1.31087 | 1.51735 | 1.43671 | 1.33743 | 1.39986 |

观察：

- **Fold 1 最难，Fold 2 次之**。Fold 1 验证 2011-11 至 2012-02，跨越年末和冬季，训练集只到 2011-10；LightGBM 的 RMSLE 为 0.5616，是五折最高。Fold 2 验证 2012-03 至 06，训练集只到 2012-02，RMSLE 为 0.4147。
- **Fold 0、3、4 较容易**。Fold 0 验证 2011-07 至 10，虽然训练数据只有 6 个月，但验证期仍在 2011 年内，LightGBM 的 RMSLE 为 0.3300；Fold 3（2012-07 至 09）最低，为 0.2550；Fold 4（2012-10 至 12）为 0.3147。训练窗口越长、验证期越接近训练分布，误差越低。
- **lgbm 在 5 折中有 4 折优于 lgbm_log**（fold 0、1、3、4），仅 fold 2 略差（0.4147 vs 0.4029）。更重要的是，lgbm 的 pooled RMSLE、MAE、RMSE、R² 全部优于 lgbm_log，因此最终选择 lgbm 没有指标上的权衡。
- **ridge 在 fold 0 明显失效（0.9438）**，因为只用 2011 年上半年 6 个月训练、外推到 2011 年下半年，线性模型对季节和增长的外推不稳定；随着训练窗口扩大，ridge 在 fold 4 降到 0.3410。**rf 的 MAE/RMSE 优于 ridge**（55.33/85.28 vs 71.78/113.62），但 RMSLE 仍落后 lgbm。
- **mean 基线 R² 为负**，因为用训练折均值预测未来月份时，需求水平和季节性都未被捕捉；fold 1 的 mean RMSLE 最高（1.5173），说明跨年 + 冬季外推对常数基线尤其困难。

### 6.3 图表与分月结果

图表：`figures/p2_model_comparison.png`（RMSLE/MAE/RMSE 三子图）、`figures/p2_fold_rmsle.png`（各折 RMSLE 折线）、`figures/p2_feature_importance.png`（最终 LightGBM Top-20 重要度）。

最佳模型 lgbm 的 OOF 分月 RMSLE（验证样本均为每月 1—19 日）：2011-07 0.2779、08 0.2580、09 0.4147、10 0.3469、11 0.2926、12 0.3609；2012-01 0.7551、02 0.6909、03 0.4131、04 0.5000、05 0.3846、06 0.3459、07 0.2519、08 0.2430、09 0.2694、10 0.3005、11 0.3464、12 0.2946。2012 年 1—2 月误差最高，因为这是跨年 + 冬季外推；2011 年 7—8 月和 2012 年 7—9 月误差最低，符合夏季需求稳定、训练窗口充分的规律。Fold 1 包含 2012-01/02，因此成为最难的一折。

## 7. 最终模型选择与拟合

按主指标 pooled OOF RMSLE，选择 **`lgbm`（Poisson 目标）**：

- CV RMSLE = 0.39973；
- 相比第二名 `lgbm_log`（0.41478）低 0.01505；
- 相比 `rf`（0.45656）低 0.05682；
- 相比 `ridge`（0.60114）低 0.20140；
- 相比 `mean`（1.40600）低 1.00627。

选择理由：主指标 RMSLE 最优；Poisson 目标直接建模非负计数，和任务性质一致；lgbm 在 pooled RMSLE、MAE、RMSE、R² 四项指标上均排名第一；fold 3/4 的 RMSLE 低于 0.32，fold 0 为 0.330，只有跨年/春季外推的 fold 1/2 较高。相比 log1p + 回归，Poisson 在低计数时段（夜间、清晨）对相对误差更友好，而 RMSLE 正是对数尺度误差。

最终 run_id：`lgbm_s42_20261101_094909_5d6e8f3390`。最终模型在全量 10,886 行上重新拟合 `FeatureBuilder.fit(all)`，使用 36 个特征，然后预测 6,493 行测试集。

最终模型参数：`objective=poisson`，`n_estimators=800`，`learning_rate=0.05`，`num_leaves=31`，`min_child_samples=30`，`subsample=0.8`，`subsample_freq=1`，`colsample_bytree=0.8`，`n_jobs=-1`，`verbose=-1`，`random_state=42`。

### 7.1 特征重要度（最终模型，gain）

数据来源：`reports/p2_feature_importance.csv`。`share` 为该特征 gain 占全部特征 gain 的比例。

| 排名 | 特征 | Gain | Split | Gain 占比 |
|---:|---|---:|---:|---:|
| 1 | hour_workingday_mean | 8,436,226.11 | 1,164 | 58.24% |
| 2 | hour_x_workingday | 2,110,145.88 | 2,880 | 14.57% |
| 3 | year | 993,101.10 | 656 | 6.86% |
| 4 | temp_x_season | 567,382.91 | 1,134 | 3.92% |
| 5 | dayofyear | 548,656.37 | 1,403 | 3.79% |
| 6 | temp | 333,087.75 | 1,050 | 2.30% |
| 7 | humidity_x_weather | 327,711.76 | 2,127 | 2.26% |
| 8 | hour_mean | 265,676.97 | 366 | 1.83% |
| 9 | atemp | 140,322.92 | 920 | 0.97% |
| 10 | hour | 84,509.62 | 633 | 0.58% |

解释：

- `hour_workingday_mean` 是最重要特征，占 58.24% gain。它把“小时 × 是否工作日”的组合目标均值直接提供给模型，强烈刻画了工作日早晚高峰与非工作日下午高峰的差异；
- `hour_x_workingday` 是同一组合的原始类别特征，占 14.57%，与聚合特征互补；
- `year` 占 6.86%，主要捕捉 2011 → 2012 的需求增长；
- 温度相关特征（`temp_x_season`、`temp`、`atemp`）合计占比约 7.2%，符合共享单车需求随气温变化的业务规律；
- `dayofyear` 及 sin/cos 编码捕捉季节性；`humidity_x_weather`、`weather`、`humidity` 捕捉天气恶劣程度的影响；
- 目标聚合特征整体占比较高，说明“小时/星期/月份的历史均值”是强信号；但这些特征在训练行上包含自身标签的组内信息，存在一定乐观偏差，详见第 9 节。

Top-1 特征 gain 远高于其他特征，图中第一根柱会明显高于其余柱；这是数据本身的信号结构，不是绘图错误。

## 8. 提交结果

文件：`reports/submission.csv`；列：`datetime,count`；行数 6,493（与 `sample_submission.csv` 一致）；时间顺序与 sample 完全一致；时间范围 2011-01-20 00:00:00 至 2012-12-31 23:00:00。

| 统计量 | 数值 |
|---|---:|
| 行数 | 6,493 |
| 最小值 | 2.2864 |
| 最大值 | 942.8630 |
| 均值 | 187.6565 |
| 中位数 | 145.9516 |
| 标准差 | 172.6926 |
| NaN 数 | 0 |
| 负值数 | 0 |

分年份：2011 年 3,223 行，均值 143.1639，范围 2.2864—569.4773；2012 年 3,270 行，均值 231.5096，范围 2.9936—942.8630。分月均值范围 56.91（2011-01）至 312.51（2012-09），符合需求的季节性和年度增长。逐月统计见 `reports/p2_submission_by_month.csv`。

已执行校验：提交行数 = 6,493；列名严格为 `datetime,count`；`datetime` 与 `sample_submission.csv` 逐行一致；`count` 无 NaN、无负值；`reports/oof/final_oof.parquet` 列严格为 `row,y,pred` 且无 NaN；`reports/p2_final_metrics.json` 可被 JSON 解析，记录 CV、fold、最终模型、提交和 OOF 信息。

## 9. 局限性与风险

### 9.1 没有随机 CV；单种子、无嵌套调参

严格使用时间切分，没有随机 KFold，也没有跨时间 shuffle。这是时间序列/未来预测任务的正确做法，但代价是：验证样本高度相关，有效样本量小于 8,196；只有 5 个验证窗口，模型选择的统计稳定性有限；不能像随机 CV 那样估计“同分布泛化误差”。此外，所有实验只跑了 `seed=42`，没有多种子平均，无法估计随机波动；超参数直接来自 `configs/base.yaml`，没有嵌套 CV 调参；在同一个 5 折上比较 5 个模型并选择最优，存在模型选择偏差；CV RMSLE 0.39973 可能略偏乐观。

### 9.2 测试集是“月内未来”，CV 无法完全模拟

这是最重要的限制：

- 训练数据只包含每月 1—19 日；
- 测试数据只包含每月 20—31 日；
- 时间 CV 的验证月份也只包含每月 1—19 日（训练数据本身没有 20 日之后的行）。

因此 CV 衡量的是“未来月份的前 19 天”误差，而测试集要求“同月后 12 天”的预测。`day`、`dayofyear`、`weekofyear` 等特征在测试集上会出现训练中未出现的取值组合；树模型只能做分段常数外推，月末误差可能高于 CV 估计。所有 CV 数字都应理解为“时间外推 + 月内前段”的参考值，而不是测试集分数的无偏估计。

### 9.3 没有目标滞后/滚动特征

`FeatureBuilder` 没有加入 `count` 的 lag、rolling mean、前一日同时段均值等特征。模型无法直接利用小时级自相关、前一天/上周同时段流量水平、突发事件后的需求惯性。这是 P1 的冻结设计，P2 没有修改。若未来允许扩展特征，优先尝试“仅使用训练期历史”的 lag/rolling 特征，并严格按时间滚动生成。此外，训练集中 `weather=4` 只有 1 行，`weather=3` 有 859 行，测试集可能出现更多恶劣天气；类别编码使用训练折类别集合，未见类别变为缺失值，LightGBM 能处理缺失但外推能力有限；`holiday` 只有 0/1 标记，没有具体节假日日历，无法区分不同节日的需求模式。

### 9.4 目标聚合特征的自包含问题

`FeatureBuilder` 的目标聚合特征（如 `hour_workingday_mean`）在 `fit` 时使用训练折所有行，包括当前行自身的 `count`。对于训练行，聚合值包含自身标签的组内贡献；对于验证/测试行，聚合值只来自训练折。这会造成训练特征分布与推理特征分布存在轻微差异，树模型可能过度依赖聚合特征；CV 验证指标本身是安全的（验证行没有参与 `fit`），但训练拟合可能偏乐观。这是 P1 的冻结实现，P2 按约定使用，未做修改。

## 10. 复现命令

以下命令均已在项目根目录 `/root/code/bike-sharing-demand` 实际运行并通过：

```bash
# 单模型时间 CV（每个模型打印 5 折与 pooled OOF，写 reports/oof/{run_id}.parquet 并追加 exp_log）
python3 -m src.models.train --config configs/base.yaml --model mean --seed 42
python3 -m src.models.train --config configs/base.yaml --model ridge --seed 42
python3 -m src.models.train --config configs/base.yaml --model rf --seed 42
python3 -m src.models.train --config configs/base.yaml --model lgbm --seed 42
python3 -m src.models.train --config configs/base.yaml --model lgbm_log --seed 42

# 五模型对比，写 p2_model_comparison.csv 与两张对比图
python3 -m src.models.train --config configs/base.yaml --compare --seed 42

# 最终全量拟合 + 测试预测，写 submission.csv、final_oof.parquet、p2_final_metrics.json、特征重要度
python3 -m src.models.predict --config configs/base.yaml --model lgbm --seed 42

# 等价的脚本入口
python3 scripts/train.py --config configs/base.yaml --model lgbm --seed 42
python3 scripts/predict.py --config configs/base.yaml --model lgbm --seed 42
```

`predict.py` 会复用对比实验中最佳 `lgbm` 的时间 CV OOF（若不存在则重新跑 CV），在全量 10,886 行上拟合最终模型，预测 6,493 行测试集，并写出第 11 节列出的产物。

## 11. 产物清单

| 文件 | 说明 |
|---|---|
| `src/models/train.py` | 时间 CV、模型实现、对比与图表 |
| `src/models/predict.py` | 全量拟合、测试预测、最终 OOF 与指标 |
| `scripts/train.py` | 训练 CLI 包装器 |
| `scripts/predict.py` | 预测 CLI 包装器 |
| `reports/exp_log.csv` | 实验日志，含 run_id、日期、模型、seed、参数哈希、CV 指标 |
| `reports/p2_model_comparison.csv` | 五模型对比表 |
| `reports/p2_final_metrics.json` | 最终模型 CV、fold、提交、OOF 元数据 |
| `reports/p2_feature_importance.csv` | 最终 LightGBM 特征重要度 |
| `reports/p2_submission_by_month.csv` | 提交结果逐月统计 |
| `reports/submission.csv` | Kaggle 提交文件（6,493 行） |
| `reports/oof/final_oof.parquet` | 最佳模型时间 CV OOF（8,196 行） |
| `reports/oof/{run_id}.parquet` | 每次 CV 运行的 OOF |
| `reports/figures/p2_model_comparison.png` | RMSLE/MAE/RMSE 模型对比 |
| `reports/figures/p2_fold_rmsle.png` | 各折 RMSLE 折线 |
| `reports/figures/p2_feature_importance.png` | Top-20 特征重要度 |
| `reports/p2_model_report.md` | 本报告 |

## 12. 总结

P2 完成了从时间 CV 到最终提交的完整闭环：在冻结的 5 折扩展窗口（time_v2）上比较了均值、Ridge、随机森林、LightGBM-Poisson 和 LightGBM-log1p；最佳模型为 **LightGBM + Poisson**，pooled OOF **RMSLE = 0.39973**、MAE = 46.586、RMSE = 73.611、R² = 0.8519；在全量 10,886 行上重新拟合，生成 6,493 行提交，预测值全部非负、无 NaN；产出可复现的 OOF（8,196 行，75.29% 覆盖）、实验日志、对比表、特征重要度和图表；并明确记录了时间外推、月内日期分布差异、无目标滞后、目标聚合自包含、单种子和模型选择偏差等限制。

后续若要进一步提升，建议优先在特征层加入严格时间滚动的 lag/rolling 特征，并在不泄漏测试标签的前提下尝试多种子平均和更细的时间折；模型层面可继续比较 Poisson 与 Tweedie 目标，或对低计数时段做专门校准。


## 11. 横向拓展

超参/收敛、特征/预处理消融、模型效率、时间漂移、多 seed 的补充实验见 `reports/extension_report.md`。与本报告直接相关的结论：

- 固定 800 棵下，`learning_rate=0.05`、`num_leaves=31`、`min_child_samples=30` 已接近最优；`num_leaves=15` 的 −0.0019 小于 seed std（0.00148），不能称显著。
- 去掉 `year` 后 RMSLE 从 0.39973 恶化到 0.49147；去掉交互特征恶化 0.0074；Poisson 优于 log1p+L2（0.39973 vs 0.41478）。
- 小模型（0.355 MB、0.81s 训练、3.5ms/1k）只损失 0.0188 RMSLE；RF 46.6 MB、151.7ms/1k，不适合部署。
- 静态 2011 模型在 2012 上加权 RMSLE 0.5402；按月扩展重训 0.3493，sliding 12 个月 0.5262。
- 3 seeds OOF RMSLE 0.39973 / 0.40177 / 0.40262，seed std 0.00148；bootstrap 95% CI [0.3910, 0.4087]。
