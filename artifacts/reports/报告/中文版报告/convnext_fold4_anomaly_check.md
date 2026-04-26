<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# ConvNeXt-Tiny Fold4 异常检查

日期：2026-04-25

## 问题

ConvNeXt-Tiny timm-aware 五折训练中，fold4 AUC 为 `0.8875`，低于 folds 1、2、3 和 5。本报告检查 fold4 是否存在数据、checkpoint、预处理或评估流程问题。

## 总结

未发现直接 bug。fold4 对 ConvNeXt-Tiny 来说确实偏低，但结果可复现，数据划分较均衡，fold4 保存的 checkpoint 与最佳验证 AUC epoch 一致，独立加载 checkpoint 后得到的指标与训练报告完全一致。

## Fold4 复评估

| 检查项 | 结果 |
| --- | --- |
| Checkpoint | `artifacts/checkpoints/convnext_tiny_timm_recipe_fold4.pt` |
| 独立验证 AUC | 0.8875 |
| 独立 Sensitivity | 0.7623 |
| 独立 Specificity | 0.8577 |
| 独立 Accuracy | 0.8267 |
| Confusion | TN 217 / FP 36 / FN 29 / TP 93 |
| 报告一致 | Yes |

这排除了简单的报告写入错误或 checkpoint 加载错误。

## 训练曲线检查

| Fold | Best Epoch | Best AUC | Last Epoch AUC | 说明 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 13 | 0.9259 | 0.9243 | 稳定高 AUC |
| 2 | 3 | 0.9301 | 0.9137 | 早期最佳，后期 AUC 降低 |
| 3 | 30 | 0.9469 | 0.9469 | 最强 fold |
| 4 | 14 | 0.8875 | 0.8837 | 平台期偏低，没有突然崩溃 |
| 5 | 10 | 0.9158 | 0.9026 | 正常运行 |

fold4 没有突然崩溃，而是从 epoch 1 的 AUC `0.8403` 提升到最佳 `0.8875`，后期保持在 `0.884` 附近，更像是验证划分更难或优化方差，而不是训练流程损坏。

## Split 分布检查

| Fold | Train Images | Val Images | Val Benign | Val Malignant | Val Malignant Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1500 | 375 | 253 | 122 | 0.3253 |
| 2 | 1500 | 375 | 254 | 121 | 0.3227 |
| 3 | 1500 | 375 | 254 | 121 | 0.3227 |
| 4 | 1500 | 375 | 253 | 122 | 0.3253 |
| 5 | 1500 | 375 | 254 | 121 | 0.3227 |

fold4 的标签分布与其他 fold 基本一致，没有发现类别不均衡问题。

## 元数据检查

| Fold | BIRADS 2 | BIRADS 3 | BIRADS 4 | BIRADS 5 | View Side Pattern | Missing BBox |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 117 | 89 | 145 | 24 | balanced left/right/single | 0 |
| 2 | 103 | 93 | 136 | 43 | balanced left/right/single | 0 |
| 3 | 108 | 98 | 148 | 21 | balanced left/right/single | 0 |
| 4 | 113 | 90 | 142 | 30 | balanced left/right/single | 0 |
| 5 | 121 | 93 | 122 | 39 | balanced left/right/single | 0 |

fold4 没有明显元数据异常，BIRADS 分布和视图侧别分布都在其他 fold 的正常范围内。

## 跨模型背景

| Fold | EfficientNetV2-S AUC | DenseNet121 AUC | ConvNeXt-Tiny AUC |
| ---: | ---: | ---: | ---: |
| 1 | 0.9248 | - | 0.9259 |
| 2 | 0.8988 | 0.8828 | 0.9301 |
| 3 | 0.9077 | 0.8849 | 0.9469 |
| 4 | 0.8891 | 0.8866 | 0.8875 |
| 5 | 0.8523 | 0.8334 | 0.9158 |

fold4 并非只对 ConvNeXt-Tiny 异常偏低；EfficientNetV2-S 和 DenseNet121 在 fold4 也落在 `0.886` 到 `0.889` 左右。这强烈说明 fold4 对当前特征分布更难，而不是 ConvNeXt 专属流水线错误。

## 结论

当前证据指向“更难的 fold / 正常折间波动”，不是数据或 checkpoint bug。

建议：

1. 保留 fold4 参与五折集成，不应在没有正式规则的情况下丢弃。
2. 如果需要进一步确认，可只用不同 seed 重训 ConvNeXt fold4，观察 AUC 是否仍接近 `0.887` 或回升到 `0.91` 以上。
3. 如果 fold4 在不同 seed 下仍然偏低，应把它当作困难验证子集用于错误案例分析，而不是作为放弃 ConvNeXt 的理由。
