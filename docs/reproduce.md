# 复现记录（本地 RTX 5060）

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB |
| 驱动 | 617.42 |
| Python | 3.10 |
| LightGBM | 4.7.0 |
| 数据 | 10,886 训练 / 6,493 测试（Kaggle 切分） |

## 一键复现

```bash
make all
# 等价于：
# make data -> make eda -> make compare -> make predict
# -> make eval -> make shap -> make badcase -> make report
# -> make ext -> make verify
```

完整跑通耗时约 **3 分 32 秒**（含 5 个横向实验）。

## 复现结果

从 raw 数据重新生成 processed、时间折、模型、OOF、submission、评估和横向实验后：

| 检查项 | 结果 |
|---|---|
| `reports/submission.csv` SHA256 | `fe77aeec46cf98f66f205a1f7fadf41102bf30bc7be63b2f239bbc8c0c4af8df` |
| 与主仓库 submission 对比 | **逐字节一致** |
| `final_oof.parquet` | 8,196 行，`row` 一致，`pred` **max abs diff = 0.0** |
| OOF RMSLE | **0.399734** |
| MAE / RMSE / R² | 46.585963 / 73.610832 / 0.851884 |
| 横向实验 A/B/D/E 数值 | 与主仓库 **完全一致**（C 的延迟因机器负载有波动） |
| `scripts/verify.py` | **52/52 通过** |
| `pytest` | **5 passed** |

复现副本位于 `/tmp/opencode/repro_bike`（临时目录，可删除）。

## 关于 RTX 5060

本项目是**表格计数回归**，主模型是 LightGBM。当前环境的 LightGBM 4.7.0 是 CPU 构建：

```text
device_type='gpu' -> LightGBMError: No OpenCL device found
```

因此训练使用 CPU，RTX 5060 只被检测到、没有被 LightGBM 使用。这是正常现象：

- 表格 GBDT 在 10,886 行数据上 CPU 已经秒级完成，GPU 不会更快；
- 如果要真正用 5060，应该换成 XGBoost/CatBoost 的 CUDA 构建，或做深度学习/图像任务；本项目没有这个必要；
- 报告里不应声称“用了 GPU 加速”，只能写“在 RTX 5060 机器上完成 CPU 复现”。

## 官方 Kaggle 数据

当前数据是用 UCI 同源数据重建的 Kaggle 切分（行数完全一致）。切换官方文件需要：

```bash
# 1. 把 kaggle.json 放到 ~/.kaggle/kaggle.json
python3 scripts/fetch_kaggle.py --replace
make data && make compare && make predict && make eval && make report
```

`scripts/fetch_kaggle.py` 会校验 10,886 / 6,493 行后再替换。
