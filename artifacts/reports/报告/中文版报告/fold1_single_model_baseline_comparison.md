# Fold1 单模型基准性能对比

> 数据边界说明：BUSBRA 用于训练、内部验证、OOF、模型选择、阈值选择和 ROI 参数选择；BUSI 是锁定外部评估集，只用于最终外部验证，不用于训练或调参。


日期：2026-04-25

## 说明

- 本报告统一评估本地已有的 fold1 分类 checkpoint，包含主线模型、调参模型和 comparison.yml 中的对比模型。
- 评估集为 BUSI；BUSI 不参与训练。
- 表中是单模型单折结果，不代表五折稳定性，也不代表最终混合集成性能。
- AlexNet 在既有 comparison 训练记录中失败，当前没有可评估 checkpoint，因此不进入排序表。

## BUSI 外部评估：默认阈值 0.50

| 排名 | 模型 | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | Precision | F1-Score | Confusion | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | ConvNeXt-Tiny timm-aware fold1 | 0.8943 | 0.7762 | 0.8741 | 0.8423 | 0.7477 | 0.7617 | TN 382 / FP 55 / FN 47 / TP 163 | recommended timm-aware recipe |
| 2 | EfficientNetV2-S sensitive balanced fold1 | 0.8803 | 0.7714 | 0.8169 | 0.8022 | 0.6694 | 0.7168 | TN 357 / FP 80 / FN 48 / TP 162 | balanced sensitivity checkpoint |
| 3 | Comparison DenseNet121 fold1 | 0.8766 | 0.4571 | 0.9771 | 0.8083 | 0.9057 | 0.6076 | TN 427 / FP 10 / FN 114 / TP 96 | comparison baseline |
| 4 | Swin-Tiny timm-aware fold1 | 0.8729 | 0.7048 | 0.8902 | 0.8300 | 0.7551 | 0.7291 | TN 389 / FP 48 / FN 62 / TP 148 | recommended timm-aware recipe |
| 5 | EfficientNetV2-S 256+aug fold1 | 0.8655 | 0.7810 | 0.8169 | 0.8053 | 0.6721 | 0.7225 | TN 357 / FP 80 / FN 46 / TP 164 | resolution + augmentation experiment |
| 6 | EfficientNetV2-S 320 fold1 | 0.8621 | 0.8667 | 0.6590 | 0.7264 | 0.5498 | 0.6728 | TN 288 / FP 149 / FN 28 / TP 182 | resolution experiment |
| 7 | Comparison EfficientNetV2-S fold1 | 0.8609 | 0.8333 | 0.7048 | 0.7465 | 0.5757 | 0.6809 | TN 308 / FP 129 / FN 35 / TP 175 | comparison baseline |
| 8 | EfficientNetV2-S 256 fold1 | 0.8599 | 0.6476 | 0.9039 | 0.8207 | 0.7640 | 0.7010 | TN 395 / FP 42 / FN 74 / TP 136 | resolution experiment |
| 9 | EfficientNetV2-S sensitive fold1 | 0.8573 | 0.9381 | 0.4645 | 0.6182 | 0.4571 | 0.6147 | TN 203 / FP 234 / FN 13 / TP 197 | sensitivity-oriented checkpoint |
| 10 | EfficientNetV2-S low-lr seed123 fold1 | 0.8508 | 0.6667 | 0.9062 | 0.8284 | 0.7735 | 0.7161 | TN 396 / FP 41 / FN 70 / TP 140 | low learning-rate seed experiment |
| 11 | EfficientNetV2-S fold1 | 0.8483 | 0.6714 | 0.8764 | 0.8099 | 0.7231 | 0.6963 | TN 383 / FP 54 / FN 69 / TP 141 | main fold1 baseline |
| 12 | Comparison ResNet18 fold1 | 0.8480 | 0.7714 | 0.8124 | 0.7991 | 0.6639 | 0.7137 | TN 355 / FP 82 / FN 48 / TP 162 | comparison baseline |
| 13 | Comparison MobileNetV3-Small fold1 | 0.8431 | 0.7143 | 0.7735 | 0.7543 | 0.6024 | 0.6536 | TN 338 / FP 99 / FN 60 / TP 150 | comparison baseline |
| 14 | Swin-Tiny initial fold1 | 0.8242 | 0.8714 | 0.5812 | 0.6754 | 0.5000 | 0.6354 | TN 254 / FP 183 / FN 27 / TP 183 | initial non-timm-aware recipe |
| 15 | Baseline ResNet18 classifier_fold1 | 0.7543 | 0.6429 | 0.7941 | 0.7450 | 0.6000 | 0.6207 | TN 347 / FP 90 / FN 75 / TP 135 | early baseline without CLAHE/TTA |
| 16 | Basic CNN fold1 | 0.7327 | 0.0429 | 0.9794 | 0.6754 | 0.5000 | 0.0789 | TN 428 / FP 9 / FN 201 / TP 9 | comparison baseline |
| 17 | ConvNeXt-Tiny initial fold1 | 0.5996 | 0.0000 | 1.0000 | 0.6754 | 0.0000 | 0.0000 | TN 437 / FP 0 / FN 210 / TP 0 | initial non-timm-aware recipe |
| 18 | Comparison VGG16 fold1 | 0.5000 | 0.0000 | 1.0000 | 0.6754 | 0.0000 | 0.0000 | TN 437 / FP 0 / FN 210 / TP 0 | comparison baseline |

## BUSI 外部评估：Youden 最优运行点

| 排名 | 模型 | AUC | 阈值 | Sens | Spec | Acc | Youden J | Precision | F1-Score | Confusion | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | ConvNeXt-Tiny timm-aware fold1 | 0.8943 | 0.48 | 0.7905 | 0.8719 | 0.8454 | 0.6623 | 0.7477 | 0.7685 | TN 381 / FP 56 / FN 44 / TP 166 | recommended timm-aware recipe |
| 2 | EfficientNetV2-S sensitive balanced fold1 | 0.8803 | 0.77 | 0.7143 | 0.8902 | 0.8331 | 0.6044 | 0.7576 | 0.7353 | TN 389 / FP 48 / FN 60 / TP 150 | balanced sensitivity checkpoint |
| 3 | Comparison DenseNet121 fold1 | 0.8766 | 0.10 | 0.6905 | 0.8902 | 0.8253 | 0.5806 | 0.7513 | 0.7196 | TN 389 / FP 48 / FN 65 / TP 145 | comparison baseline |
| 4 | Swin-Tiny timm-aware fold1 | 0.8729 | 0.16 | 0.7429 | 0.8719 | 0.8300 | 0.6147 | 0.7358 | 0.7393 | TN 381 / FP 56 / FN 54 / TP 156 | recommended timm-aware recipe |
| 5 | EfficientNetV2-S 256+aug fold1 | 0.8655 | 0.47 | 0.7905 | 0.8124 | 0.8053 | 0.6028 | 0.6694 | 0.7249 | TN 355 / FP 82 / FN 44 / TP 166 | resolution + augmentation experiment |
| 6 | EfficientNetV2-S 320 fold1 | 0.8621 | 0.90 | 0.8048 | 0.7872 | 0.7929 | 0.5919 | 0.6450 | 0.7161 | TN 344 / FP 93 / FN 41 / TP 169 | resolution experiment |
| 7 | Comparison EfficientNetV2-S fold1 | 0.8609 | 0.87 | 0.7714 | 0.8169 | 0.8022 | 0.5884 | 0.6694 | 0.7168 | TN 357 / FP 80 / FN 48 / TP 162 | comparison baseline |
| 8 | EfficientNetV2-S 256 fold1 | 0.8599 | 0.31 | 0.7143 | 0.8696 | 0.8192 | 0.5839 | 0.7246 | 0.7194 | TN 380 / FP 57 / FN 60 / TP 150 | resolution experiment |
| 9 | EfficientNetV2-S sensitive fold1 | 0.8573 | 0.90 | 0.8286 | 0.7048 | 0.7450 | 0.5334 | 0.5743 | 0.6784 | TN 308 / FP 129 / FN 36 / TP 174 | sensitivity-oriented checkpoint |
| 10 | EfficientNetV2-S low-lr seed123 fold1 | 0.8508 | 0.27 | 0.7810 | 0.8192 | 0.8068 | 0.6002 | 0.6749 | 0.7241 | TN 358 / FP 79 / FN 46 / TP 164 | low learning-rate seed experiment |
| 11 | EfficientNetV2-S fold1 | 0.8483 | 0.24 | 0.7571 | 0.8169 | 0.7975 | 0.5741 | 0.6653 | 0.7082 | TN 357 / FP 80 / FN 51 / TP 159 | main fold1 baseline |
| 12 | Comparison ResNet18 fold1 | 0.8480 | 0.63 | 0.7524 | 0.8398 | 0.8114 | 0.5922 | 0.6930 | 0.7215 | TN 367 / FP 70 / FN 52 / TP 158 | comparison baseline |
| 13 | Comparison MobileNetV3-Small fold1 | 0.8431 | 0.72 | 0.6857 | 0.8444 | 0.7929 | 0.5301 | 0.6792 | 0.6825 | TN 369 / FP 68 / FN 66 / TP 144 | comparison baseline |
| 14 | Swin-Tiny initial fold1 | 0.8242 | 0.57 | 0.7571 | 0.7574 | 0.7573 | 0.5146 | 0.6000 | 0.6695 | TN 331 / FP 106 / FN 51 / TP 159 | initial non-timm-aware recipe |
| 15 | Baseline ResNet18 classifier_fold1 | 0.7543 | 0.50 | 0.6429 | 0.7941 | 0.7450 | 0.4369 | 0.6000 | 0.6207 | TN 347 / FP 90 / FN 75 / TP 135 | early baseline without CLAHE/TTA |
| 16 | Basic CNN fold1 | 0.7327 | 0.38 | 0.6048 | 0.7941 | 0.7326 | 0.3988 | 0.5853 | 0.5948 | TN 347 / FP 90 / FN 83 / TP 127 | comparison baseline |
| 17 | ConvNeXt-Tiny initial fold1 | 0.5996 | 0.10 | 1.0000 | 0.0000 | 0.3246 | 0.0000 | 0.3246 | 0.4901 | TN 0 / FP 437 / FN 0 / TP 210 | initial non-timm-aware recipe |
| 18 | Comparison VGG16 fold1 | 0.5000 | 0.10 | 1.0000 | 0.0000 | 0.3246 | 0.0000 | 0.3246 | 0.4901 | TN 0 / FP 437 / FN 0 / TP 210 | comparison baseline |

## 结论

- Fold1 单模型里，BUSI AUC 最高的是 ConvNeXt-Tiny timm-aware fold1，AUC `0.8943`。
- timm-aware 配方明显修正了 ConvNeXt-Tiny 和 Swin-Tiny 的早期训练/预处理问题。
- 单个 fold1 只适合做模型基础潜力参考；最终选择仍应优先看五折和外部评估。
