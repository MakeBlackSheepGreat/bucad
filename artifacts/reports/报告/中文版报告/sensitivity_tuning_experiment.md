<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# Sensitivity 调优实验

日期：2026-04-24

## 目标

在不使用 BUSI 训练的前提下，改善 `224` EfficientNetV2-S classifier 的 fold-1 Sensitivity 偏低问题。

## 方法

| Experiment | Config | Main Change | Checkpoint Selection |
| --- | --- | --- | --- |
| Baseline | `configs/classifier/efficientnetv2_s.yml` | original 224 training | final epoch |
| Sensitive | `configs/classifier/efficientnetv2_s_sensitive.yml` | malignant class weight 1.35 | threshold 0.24 sensitivity-weighted score |
| Balanced sensitive | `configs/classifier/efficientnetv2_s_sensitive_balanced.yml` | malignant class weight 1.20 | threshold 0.24 score with specificity constraint |

## Fold-1 验证结果

| Experiment | Best Epoch | AUC | Sensitivity @0.50 | Specificity @0.50 | Accuracy @0.50 | Sensitivity @0.24 | Specificity @0.24 | Accuracy @0.24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 0.9248 | 0.6148 | 0.9565 | 0.8453 | - | - | - |
| Sensitive | 18 | 0.8922 | 0.9016 | 0.7075 | 0.7707 | 0.9262 | 0.6482 | 0.7387 |
| Balanced sensitive | 27 | 0.9170 | 0.7541 | 0.8775 | 0.8373 | 0.8033 | 0.8696 | 0.8480 |

## 决策

- 不基于单 fold 结果替换冻结的五折 runtime ensemble。
- aggressive sensitive config 大幅提升 Sensitivity，但牺牲过多 Specificity 和 AUC。
- balanced sensitive config 是更好的后续候选：默认阈值 Sensitivity 从 `0.6148` 提升到 `0.7541`，同时 AUC 接近 baseline（`0.9170` vs `0.9248`）。
- 如果训练时间充足，可对 `configs/classifier/efficientnetv2_s_sensitive_balanced.yml` 跑完整五折，并在 BUSI 上评估 ensemble 后再考虑 runtime 切换。
