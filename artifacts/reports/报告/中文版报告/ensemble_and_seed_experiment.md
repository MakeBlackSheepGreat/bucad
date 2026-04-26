<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 集成与 Seed 实验

日期：2026-04-24

## 目标

测试两个优先级较高的优化方向：

1. 异构模型集成。
2. 使用不同 seed 的低学习率重训练。

BUSI 只用于外部评估，不参与训练。

## BUSI 上的混合集成结果

| Ensemble | AUC | Sensitivity @0.50 | Specificity @0.50 | Accuracy @0.50 | Best Threshold | Best Sensitivity | Best Specificity | Best Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNetV2-S 5-fold + TTA | 0.8997 | 0.6619 | 0.9382 | 0.8485 | 0.33 | 0.8000 | 0.8764 | 0.8516 |
| EfficientNetV2-S 5-fold + DenseNet121 fold1 weight 0.25 | 0.9011 | 0.6476 | 0.9428 | 0.8470 | 0.32 | 0.8000 | 0.8764 | 0.8516 |
| EfficientNetV2-S 5-fold + DenseNet121 fold1 weight 0.50 | 0.9021 | 0.6476 | 0.9428 | 0.8470 | 0.30 | 0.8095 | 0.8719 | 0.8516 |
| EfficientNetV2-S 5-fold + DenseNet121 5-fold weight 0.25 | 0.9037 | 0.6286 | 0.9474 | 0.8439 | 0.29 | 0.8095 | 0.8719 | 0.8516 |
| EfficientNetV2-S 5-fold + DenseNet121 5-fold weight 0.50 | 0.9050 | 0.5905 | 0.9519 | 0.8346 | 0.28 | 0.8095 | 0.8696 | 0.8501 |
| EfficientNetV2-S 5-fold + DenseNet121 5-fold weight 0.63 | 0.9052 | 0.5810 | 0.9542 | 0.8331 | 0.27 | 0.8095 | 0.8719 | 0.8516 |
| EfficientNetV2-S 5-fold + DenseNet121 5-fold weight 1.00 | 0.9051 | 0.5714 | 0.9611 | 0.8346 | 0.23 | 0.8190 | 0.8581 | 0.8454 |
| EfficientNetV2-S fold1 + DenseNet121 fold1 | 0.8767 | 0.5524 | 0.9497 | 0.8207 | 0.15 | - | - | - |
| EfficientNetV2-S fold1 + ResNet18 fold1 | 0.8612 | 0.6857 | 0.8879 | 0.8223 | 0.29 | - | - | - |
| EfficientNetV2-S fold1 + MobileNetV3 fold1 | 0.8629 | 0.7095 | 0.8764 | 0.8223 | 0.38 | - | - | - |

## 低学习率 Seed 结果

| Experiment | Config | Best Epoch | AUC | Sensitivity @0.50 | Specificity @0.50 | Accuracy @0.50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline EfficientNetV2-S fold1 | `configs/classifier/efficientnetv2_s.yml` | - | 0.9248 | 0.6148 | 0.9565 | 0.8453 |
| LR 1e-4 seed 123 fold1 | `configs/classifier/efficientnetv2_s_lr1e4_seed123.yml` | 27 | 0.8503 | 0.6639 | 0.8656 | 0.8000 |

## 决策

- 眼前最有效的优化是异构集成。完整 DenseNet121 五折补充分支将 BUSI AUC 从 `0.8997` 提升到 `0.9052`。
- 单折异构集成弱于当前五折 EfficientNetV2-S ensemble，因此混合应作为已有 ensemble 的小权重补充，而不是替代。
- 首次低学习率 seed 运行没有帮助；除非有时间做多 seed 和更强 checkpoint 选择，否则不继续大规模 seed sweep。
- 建议下一步保留 mixed ensemble 作为 runtime candidate，只有当其他 seed 或结构实验超过 `0.9052` BUSI AUC baseline 时再替换。
