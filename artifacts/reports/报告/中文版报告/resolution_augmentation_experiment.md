<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 分辨率与增强实验

日期：2026-04-24

## 目标

评估分类器是否应从冻结的 `224` pipeline 切换到更高分辨率 `256` pipeline，以及额外 ultrasound-safe augmentation 是否能改善 fold-1 验证表现。

## 配置

| Experiment | Config | Image Size | Extra Augmentation |
| --- | --- | ---: | --- |
| Baseline | `configs/classifier/efficientnetv2_s.yml` | 224 | horizontal flip |
| Resolution only | `configs/classifier/efficientnetv2_s_256.yml` | 256 | horizontal flip |
| Resolution + mild aug | `configs/classifier/efficientnetv2_s_256_aug.yml` | 256 | horizontal flip, rotation, brightness, contrast, scale crop |
| Resolution 320 | `configs/classifier/efficientnetv2_s_320.yml` | 320 | horizontal flip |

## Fold-1 结果

| Experiment | AUC | Sensitivity | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 224 baseline | 0.9248 | 0.6148 | 0.9565 | 0.8453 |
| 256 resolution only | 0.8994 | 0.7213 | 0.9328 | 0.8640 |
| 256 mild augmentation | 0.8811 | 0.7541 | 0.8379 | 0.8107 |
| 320 resolution only | 0.8903 | 0.7541 | 0.8735 | 0.8347 |

## 决策

- 暂不使用任何 `256` 实验替换冻结 runtime model。
- `256` resolution-only 提升 fold-1 Sensitivity 和 Accuracy，但相较 `224` baseline 明显降低 AUC。
- 额外 mild augmentation 进一步提高 Sensitivity，但降低 AUC、Specificity 和 Accuracy，说明当前增强强度不适合作为最终模型方向。
- `320` resolution-only 没有恢复 `256` 中出现的 AUC 损失；相比冻结 `224` baseline，它提高 Sensitivity，但降低 AUC、Specificity 和 Accuracy。
- 保留新 configs 和 augmentation code 以便后续复现实验，但 `configs/inference/demo.yml` 继续使用现有 EfficientNetV2-S 五折 runtime ensemble。

## 后续选项

- 尝试 `256` + 更低学习率，如 `0.0001`。
- 尝试 `256` + early stopping 或 best-validation checkpoint saving。
- 只有当 BUSI 外部评估确认有收益时，再尝试当前 `224` fold 模型与部分 `256` 模型混合集成。
