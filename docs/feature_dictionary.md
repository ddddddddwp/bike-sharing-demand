# Bike Sharing Demand 特征字典（P1）

> 对应代码：`src/features/build.py`（接口冻结）
> 数据：`data/processed/train.parquet`、`data/processed/test.parquet`
> 目标：`count`（每小时租车总数），主指标 RMSLE
> 本文档描述当前实现，不改变 `FeatureBuilder` 的公开接口。

## 1. 输入字段

| 字段 | 类型 | 含义 | 备注 |
|---|---|---|---|
| `datetime` | datetime | 小时时间戳 | 2011-01-01 00:00 至 2012-12-31 23:00；训练/测试按日期切分 |
| `season` | int | 季节，1=春、2=夏、3=秋、4=冬 | 类别 |
| `holiday` | 0/1 | 是否法定假日 | 与 `workingday` 不完全互斥 |
| `workingday` | 0/1 | 是否工作日 | 0 包含周末和假日 |
| `weather` | int | 天气等级，1=晴、2=雾/多云、3=小雨雪、4=大雨雪 | 类别；4 类训练集仅 1 行 |
| `temp` | float | 归一化温度 | UCI 原始口径，约 `(摄氏温度-8)/47` |
| `atemp` | float | 归一化体感温度 | 与 `temp` 高度相关 |
| `humidity` | float | 归一化湿度 | 0–1 |
| `windspeed` | float | 归一化风速 | 0–1 |
| `casual` | int | 非注册用户租车数 | **泄漏字段，禁止入模** |
| `registered` | int | 注册用户租车数 | **泄漏字段，禁止入模** |
| `count` | int | 总租车数 = `casual + registered` | 目标，仅训练集有 |

`test.csv` 只有前 9 个字段（不含 `casual`、`registered`、`count`）。原始数据没有缺失值，但存在**缺失小时**：训练集 456 个自然日、10,886 行，相对完整 24 小时网格少 58 个小时；测试集 275 个自然日、6,493 行，少 107 个小时。当前流程不插值、不重采样，按原始时间戳建模和预测。

## 2. 特征构建流程

```python
builder = FeatureBuilder(use_target_aggregates=True)
builder.fit(train_df)                 # 只能看到当前训练折
X_train = builder.transform(train_df)
X_valid = builder.transform(valid_df)
```

- `fit` 只从传入的训练行计算目标聚合；`transform` 不读取目标。
- `transform` 删除 `datetime`、`count`、`casual`、`registered`。
- `use_target_aggregates=True` 时输出 **36** 个特征；`False` 时输出 **29** 个特征。
- 所有预测必须 `clip(pred, 0, None)`；RMSLE 在 `log1p(count)` 空间计算，模型本身可以在 raw count（如 Poisson）或 `log1p` 目标上训练。

## 3. 特征分组

### 3.1 原始数值/类别特征（8 个）

`season`、`holiday`、`workingday`、`weather`、`temp`、`atemp`、`humidity`、`windspeed`。

其中 `season`、`weather` 以及派生的 `weekday`、`month` 被转换为 pandas `category` dtype；`hour_x_workingday` 为字符串类别（object），树模型使用时应显式转 category 或做 one-hot。其余为数值。

### 3.2 日历特征（7 个）

| 特征 | 说明 | dtype |
|---|---|---|
| `year` | 2011 / 2012 | int16 |
| `month` | 1–12 | category |
| `day` | 月内日期 1–31 | int8 |
| `hour` | 0–23 | int8 |
| `weekday` | 周一=0 … 周日=6 | category |
| `dayofyear` | 1–366 | int16 |
| `weekofyear` | ISO 周 | int8 |

注意：Kaggle 训练集只含每月 1–19 日，测试集是 20 日–月末。`day` 在训练中只出现 1–19，测试的 20–31 对树模型是**外推区间**；它不泄漏目标，但泛化价值有限，属于已知分布偏移。

### 3.3 二元标记（3 个）

| 特征 | 规则 |
|---|---|
| `is_weekend` | `weekday >= 5` |
| `is_rush_hour` | `hour in {7,8,9,17,18,19}` |
| `is_night` | `hour in {0,1,2,3,4,5}` |

### 3.4 周期编码（8 个）

用正弦/余弦保留时间的周期性，避免 23 点与 0 点在数值上相距很远：

- `hour_sin`、`hour_cos`：周期 24
- `weekday_sin`、`weekday_cos`：周期 7
- `month_sin`、`month_cos`：周期 12
- `dayofyear_sin`、`dayofyear_cos`：周期 365.25

### 3.5 交互特征（3 个）

| 特征 | 构造 | 说明 |
|---|---|---|
| `hour_x_workingday` | `str(hour) + "_" + str(workingday)` | 24×2 组合，捕捉工作日通勤双峰与周末午后单峰 |
| `temp_x_season` | `temp * season` | 温度效应随季节变化 |
| `humidity_x_weather` | `humidity * weather` | 湿度效应随天气等级变化 |

### 3.6 目标聚合特征（7 个，可关闭）

当 `use_target_aggregates=True` 时增加：

| 特征 | 分组键 |
|---|---|
| `hour_mean` | `hour` |
| `weekday_mean` | `weekday` |
| `month_mean` | `month` |
| `season_mean` | `season` |
| `weather_mean` | `weather` |
| `workingday_mean` | `workingday` |
| `hour_workingday_mean` | `hour_x_workingday` |

计算方式（平滑均值）：

```text
global = mean(y_train)
agg(g) = (sum(y_train | group=g) + m * global) / (count(g) + m)
m = 20
```

- 只在 `fit` 的当前训练行上计算，验证/测试行通过 `transform` 映射。
- 未见过的分组回退到 `global`；因此验证集中未来月份 `month_mean` 会回退，不会读取验证目标。
- `hour_workingday_mean` 是 48 个格子，平滑后比单独 `hour_mean` 更细。
- 目标聚合是**折内编码**：P2 做 CV 时必须每个 fold 重新 `fit`，不能在全量训练集上先 fit 再切折；否则验证折信息进入编码，CV 会偏乐观。
- 目标聚合在训练行上是样本内编码，树模型可能过拟合；本项目的 Ridge 诊断显示 `ridge` 与 `ridge_target_enc` 的 OOF RMSLE 几乎相同（0.4740 vs 0.4739），因此没有证据表明它显著提升线性基线。

## 4. 泄漏审计

| 检查项 | 结论 |
|---|---|
| `count` 是否入模 | 否，`DROP` 包含 `count` |
| `casual` / `registered` 是否入模 | 否，`DROP` 包含两者 |
| `casual + registered == count` | 训练集 10,886/10,886 行成立（100%） |
| 与 `count` 的 Pearson 相关 | `registered` 0.9709，`casual` 0.6904 |
| `datetime` 是否入模 | 否，时间信息只以日历/周期特征进入 |
| 目标聚合是否使用验证目标 | 否；`fit` 只接收训练行，`transform` 只做映射 |
| 目标聚合是否全量预计算 | 否；P2 必须按 fold 调用 `fit` |
| 随机切分 | 禁止；见第 5 节 |

`casual` 和 `registered` 是目标的两个加数，单独使用任意一个都能近乎复现 `count`，属于确定性泄漏；即使只用一个也会让 RMSLE 虚低。本项目从特征层直接删除，不做任何“只用一个”的折中。

## 5. 为什么不能用随机 KFold

1. **时间依赖**：相邻小时、相邻日期的租赁量高度相关。随机切分会把同一时间段的小时同时放进训练和验证，模型可以“记住”邻近样本，CV 虚高。
2. **季节与增长**：数据有强季节性和 2012 年相对 2011 年的增长（训练集年均 `count` 144.22 → 238.56，+65.4%）。随机切分让未来信息进入训练，无法模拟“用过去预测未来”。
3. **目标聚合泄漏**：若先在全量训练集上计算 `hour_mean` 等，再随机切分，验证折目标已经进入特征；即使按折重算，随机折仍会把未来月份放进训练。
4. **与比赛口径不一致**：Kaggle 训练集是每月 1–19 日，测试集是 20 日–月末；本地必须用时间切分模拟这一“未来日期”场景。

因此使用 `splits/time_folds.json` 的 5 个 expanding-window 折：训练窗口从 2011-01 起逐折扩大，验证窗口是后续月份；折内训练时间严格早于验证时间，训练/验证索引无交集。

## 6. 接口稳定性与改动记录

- `FeatureBuilder.__init__(use_target_aggregates=True, smooth=20.0)`、`fit`、`transform`、`feature_names` 均保持原签名和返回约定。
- **未新增特征**，未改变输出列顺序；本次 P1 只补充数据管线、EDA、基线、文档。
- 已知实现细节：`hour_x_workingday` 仍是 object dtype；若 P2 的模型只接受数值矩阵，需要在模型侧做 one-hot 或 `astype("category")`。这属于调用侧适配，不改变 builder 接口。
- `feature_names` 在 `transform` 后填充；若在 `transform` 之前访问，返回空列表，这是原有行为。

## 7. 快速核对

```python
from src.features.build import FeatureBuilder

b = FeatureBuilder(use_target_aggregates=True).fit(train_df)
X = b.transform(train_df)
assert "count" not in X.columns
assert "casual" not in X.columns
assert "registered" not in X.columns
assert "datetime" not in X.columns
print(X.shape)          # (n_rows, 36)
print(b.feature_names)
```
