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

因此主模型训练使用 CPU。为了验证 5060 是否有用，额外装了 `xgboost-cu12==3.2.0`，做了同一 `time_v2` 折的 CPU vs CUDA 对比：

| 设备/模型 | CV RMSLE | 训练秒数（5 折） | 推理延迟 |
|---|---:|---:|---:|
| XGBoost CPU | 0.40348 | 4.87 | 3.50 ms/1k |
| XGBoost CUDA（5060） | 0.40674 | 13.29 | 6.20 ms/1k |
| LightGBM CPU（主模型） | 0.39973 | 4.77 | 8.27 ms/1k |

**在 10,886 行上，GPU 反而慢约 2.7 倍**：CUDA 启动/传输/同步开销占主导。  
另做 200k 行合成压力测试：CPU 3.6–5.0s，CUDA 2.3s，GPU 约 1.6–2.2 倍加速（受负载波动）。  
所以结论是：**5060 可用，但本项目数据规模太小，主模型继续用 CPU；GPU 只在数据量上来后才有收益。**

复现 GPU 对比：

```bash
pip install -r requirements-gpu.txt
make gpu
```

报告里不要写“主模型用了 GPU 加速”，只能写“在 RTX 5060 机器上完成 CPU 主流程 + 额外 GPU 对比实验”。

## 官方 Kaggle 数据

当前数据是用 UCI 同源数据重建的 Kaggle 切分（行数完全一致）。切换官方文件需要：

```bash
# 1. 把 kaggle.json 放到 ~/.kaggle/kaggle.json
python3 scripts/fetch_kaggle.py --replace
make data && make compare && make predict && make eval && make report
```

`scripts/fetch_kaggle.py` 会校验 10,886 / 6,493 行后再替换。
