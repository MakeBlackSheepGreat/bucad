<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 正式实验前置条件

日期：2026-04-24

## 范围

本清单确认项目是否已经具备运行正式实验的条件，包括 EfficientNetV2-S 五折训练、完整分类器对比、BUSI 外部评估和 release freeze。

## 环境

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| Conda environment | PASS | `conda run -n BUCAD python --version` 为 Python 3.11.15 |
| Required Python packages | PASS | `check_env.py` 报告 required 和 optional packages 可用 |
| CUDA runtime | PASS | PyTorch reports CUDA available |
| GPU | PASS | NVIDIA GeForce RTX 5060 Laptop GPU |
| Regression baseline | PASS | `check_all.py` 通过 26 个测试 |

## 数据可用性

| Dataset | Required Role | Local Path | Status | Evidence |
| --- | --- | --- | --- | --- |
| BUSBRA | Training and internal validation only | `训练集/BUSBRA` | PASS | `训练集` 下检测到 3,754 个文件 |
| BUSI | External evaluation and demo validation only | `测试集/Dataset_BUSI_with_GT` | PASS | `测试集` 下检测到 1,578 个文件 |
| BUSBRA 5-fold split | Case-level split assignments | `artifacts/reports/busbra_5fold_splits.csv` | PASS | 1,875 samples，1,064 unique cases，`leakage_detected=false` |

## 当前 Checkpoint

| Checkpoint | Status | Notes |
| --- | --- | --- |
| `artifacts/checkpoints/classifier_fold1.pt` | AVAILABLE | 现有 smoke/formal seed classifier checkpoint |
| `artifacts/checkpoints/segmenter_fold1.pt` | AVAILABLE | 现有 segmentation checkpoint，用于 visual-evidence flow |
| `artifacts/checkpoints/efficientnetv2_s_fold{fold}.pt` | PENDING | 由 T059 五折 EfficientNetV2-S 训练生成 |

## 当前报告

| Report | Status | Notes |
| --- | --- | --- |
| `artifacts/reports/train_cls_fold1.json` | AVAILABLE | 现有分类器训练证据 |
| `artifacts/reports/busi_eval.json` | AVAILABLE | 现有 BUSI 评估证据 |
| `artifacts/reports/comparison_results.json` | PARTIAL | 当前文件是 dry-run/smoke comparison，不是完整模型对比 |
| `artifacts/reports/threshold_analysis.md` | AVAILABLE | 现有 threshold-analysis note |
| `artifacts/reports/final_validation.md` | AVAILABLE | 现有 regression/demo validation evidence |

## 正式实验就绪情况

- PASS：数据边界已准备好，BUSBRA 只用于训练/内部验证，BUSI 只用于外部评估/demo 验证。
- PASS：病例级 split 文件存在并记录无泄漏。
- PASS：BUCAD 环境可导入 PyTorch、OpenCV、timm、segmentation_models_pytorch 和 Gradio。
- PASS：CUDA 可用于长时间正式训练。
- PENDING：仍需生成最终 EfficientNetV2-S 五折 checkpoint 和报告。
- PENDING：完整模型对比需要替换当前 dry-run comparison report。
- PENDING：最终 classifier 或 ensemble 冻结后，应重新运行最终 BUSI evaluation。

## 下一步命令

运行 `artifacts/reports/formal_experiment_commands.md` 中记录的命令。
