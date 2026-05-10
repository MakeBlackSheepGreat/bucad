# 原生/推荐单模型重测报告

日期：2026-04-26

## 说明

- 本报告重测项目中可用的单折原生或推荐协议模型，不包含后续低学习率、分辨率、敏感性、seed、ROI、OOF 或集成调优变体。
- 评估输出统一包含 AUC、Accuracy、Recall/Sensitivity、Precision、Specificity 和 F1-Score。
- `AlexNet` 在既有训练记录中失败，当前没有可评估 checkpoint，因此仍不进入表格。
- `ConvNeXt-Small` 是后续升级实验补充模型，不属于最初四模型主线筛选，但一并列出供参考。

## 默认阈值 0.50 重测结果

| 排名 | 模型 | 阈值 | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Confusion | 协议 | 备注 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | ConvNeXt-Small timm recipe fold1 | 0.50 | 0.8947 | 0.8284 | 0.7667 | 0.7220 | 0.8581 | 0.7436 | TN 375 / FP 62 / FN 49 / TP 161 | 224 + CLAHE + timm mean/std + bicubic + crop 0.95 + hflip TTA | 后续升级验证，不进入默认主线 |
| 2 | ConvNeXt-Tiny timm recipe fold1 | 0.50 | 0.8943 | 0.8423 | 0.7762 | 0.7477 | 0.8741 | 0.7617 | TN 382 / FP 55 / FN 47 / TP 163 | 224 + CLAHE + timm mean/std + bicubic + crop 0.95 + hflip TTA | ConvNeXt 推荐配方 |
| 3 | DenseNet121 fold1 | 0.50 | 0.8766 | 0.8083 | 0.4571 | 0.9057 | 0.9771 | 0.6076 | TN 427 / FP 10 / FN 114 / TP 96 | 224 + CLAHE + hflip TTA | Dense connectivity candidate |
| 4 | Swin-Tiny timm recipe fold1 | 0.50 | 0.8729 | 0.8300 | 0.7048 | 0.7551 | 0.8902 | 0.7291 | TN 389 / FP 48 / FN 62 / TP 148 | 224 + CLAHE + timm mean/std + bicubic + crop 0.90 + hflip TTA | Swin 推荐配方 |
| 5 | EfficientNetV2-S fold1 | 0.50 | 0.8609 | 0.7465 | 0.8333 | 0.5757 | 0.7048 | 0.6809 | TN 308 / FP 129 / FN 35 / TP 175 | 224 + CLAHE + hflip TTA | EfficientNetV2 candidate |
| 6 | EfficientNetV2-S main fold1 | 0.50 | 0.8483 | 0.8099 | 0.6714 | 0.7231 | 0.8764 | 0.6963 | TN 383 / FP 54 / FN 69 / TP 141 | 224 + CLAHE + hflip TTA | 后续五折主线 EfficientNet 分支的 fold1 |
| 7 | ResNet18 fold1 | 0.50 | 0.8480 | 0.7991 | 0.7714 | 0.6639 | 0.8124 | 0.7137 | TN 355 / FP 82 / FN 48 / TP 162 | 224 + CLAHE + hflip TTA | ResNet baseline |
| 8 | MobileNetV3-Small fold1 | 0.50 | 0.8431 | 0.7543 | 0.7143 | 0.6024 | 0.7735 | 0.6536 | TN 338 / FP 99 / FN 60 / TP 150 | 224 + CLAHE + hflip TTA | MobileNet lightweight baseline |
| 9 | Swin-Tiny initial fold1 | 0.50 | 0.8242 | 0.6754 | 0.8714 | 0.5000 | 0.5812 | 0.6354 | TN 254 / FP 183 / FN 27 / TP 183 | 224 + CLAHE + hflip TTA, no timm recipe | 早期非 timm-aware Swin |
| 10 | Early ResNet18 classifier fold1 | 0.50 | 0.7543 | 0.7450 | 0.6429 | 0.6000 | 0.7941 | 0.6207 | TN 347 / FP 90 / FN 75 / TP 135 | 224 + identity TTA | 早期 ResNet18 baseline |
| 11 | Basic CNN fold1 | 0.50 | 0.7327 | 0.6754 | 0.0429 | 0.5000 | 0.9794 | 0.0789 | TN 428 / FP 9 / FN 201 / TP 9 | 224 + CLAHE + hflip TTA | 非预训练轻量 CNN baseline |
| 12 | ConvNeXt-Tiny initial fold1 | 0.50 | 0.5996 | 0.6754 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | TN 437 / FP 0 / FN 210 / TP 0 | 224 + CLAHE + hflip TTA, no timm recipe | 早期非 timm-aware ConvNeXt |
| 13 | VGG16 fold1 | 0.50 | 0.5000 | 0.6754 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | TN 437 / FP 0 / FN 210 / TP 0 | 224 + CLAHE + hflip TTA | VGG baseline |

## Youden 最优运行点

| 排名 | 模型 | 阈值 | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Youden J | Confusion |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | ConvNeXt-Small timm recipe fold1 | 0.20 | 0.8947 | 0.8223 | 0.8238 | 0.6892 | 0.8215 | 0.7505 | 0.6453 | TN 359 / FP 78 / FN 37 / TP 173 |
| 2 | ConvNeXt-Tiny timm recipe fold1 | 0.48 | 0.8943 | 0.8454 | 0.7905 | 0.7477 | 0.8719 | 0.7685 | 0.6623 | TN 381 / FP 56 / FN 44 / TP 166 |
| 3 | DenseNet121 fold1 | 0.10 | 0.8766 | 0.8253 | 0.6905 | 0.7513 | 0.8902 | 0.7196 | 0.5806 | TN 389 / FP 48 / FN 65 / TP 145 |
| 4 | Swin-Tiny timm recipe fold1 | 0.16 | 0.8729 | 0.8300 | 0.7429 | 0.7358 | 0.8719 | 0.7393 | 0.6147 | TN 381 / FP 56 / FN 54 / TP 156 |
| 5 | EfficientNetV2-S fold1 | 0.87 | 0.8609 | 0.8022 | 0.7714 | 0.6694 | 0.8169 | 0.7168 | 0.5884 | TN 357 / FP 80 / FN 48 / TP 162 |
| 6 | EfficientNetV2-S main fold1 | 0.24 | 0.8483 | 0.7975 | 0.7571 | 0.6653 | 0.8169 | 0.7082 | 0.5741 | TN 357 / FP 80 / FN 51 / TP 159 |
| 7 | ResNet18 fold1 | 0.63 | 0.8480 | 0.8114 | 0.7524 | 0.6930 | 0.8398 | 0.7215 | 0.5922 | TN 367 / FP 70 / FN 52 / TP 158 |
| 8 | MobileNetV3-Small fold1 | 0.72 | 0.8431 | 0.7929 | 0.6857 | 0.6792 | 0.8444 | 0.6825 | 0.5301 | TN 369 / FP 68 / FN 66 / TP 144 |
| 9 | Swin-Tiny initial fold1 | 0.57 | 0.8242 | 0.7573 | 0.7571 | 0.6000 | 0.7574 | 0.6695 | 0.5146 | TN 331 / FP 106 / FN 51 / TP 159 |
| 10 | Early ResNet18 classifier fold1 | 0.50 | 0.7543 | 0.7450 | 0.6429 | 0.6000 | 0.7941 | 0.6207 | 0.4369 | TN 347 / FP 90 / FN 75 / TP 135 |
| 11 | Basic CNN fold1 | 0.38 | 0.7327 | 0.7326 | 0.6048 | 0.5853 | 0.7941 | 0.5948 | 0.3988 | TN 347 / FP 90 / FN 83 / TP 127 |
| 12 | ConvNeXt-Tiny initial fold1 | 0.10 | 0.5996 | 0.3246 | 1.0000 | 0.3246 | 0.0000 | 0.4901 | 0.0000 | TN 0 / FP 437 / FN 0 / TP 210 |
| 13 | VGG16 fold1 | 0.10 | 0.5000 | 0.3246 | 1.0000 | 0.3246 | 0.0000 | 0.4901 | 0.0000 | TN 0 / FP 437 / FN 0 / TP 210 |

## 四个候选模型的由来

- 初筛阶段，DenseNet121、EfficientNetV2-S、ConvNeXt-Tiny 和 Swin-Tiny 的单折表现、结构互补性或后续可优化空间最好，因此被选为后续 BUCAD 五折训练和混合集成的主要候选。
- ConvNeXt-Tiny 在引入 timm-aware 预处理后从早期失败状态显著提升，是后续主模型的核心候选。
- EfficientNetV2-S 与 DenseNet121 在 CNN 家族中表现稳定，分别提供较好的综合性能和较高 Precision/Specificity 参考。
- Swin-Tiny 单折不如 ConvNeXt-Tiny，但作为 Transformer 风格候选具备互补性，因此进入后续五折与混合集成测试。
- 后续正式 demo 没有直接采用四模型全集成，而是根据五折、TTA、OOF、ROI 和部署复杂度进一步收敛到 `ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate`。

## 输出文件

- `artifacts/reports/busi_single_convnext_small_timm_recipe_fold1.json`
- `artifacts/reports/busi_single_convnext_tiny_timm_recipe_fold1.json`
- `artifacts/reports/busi_single_comparison_densenet121_fold1.json`
- `artifacts/reports/busi_single_swin_tiny_timm_recipe_fold1.json`
- `artifacts/reports/busi_single_comparison_tf_efficientnetv2_s_fold1.json`
- `artifacts/reports/busi_single_efficientnetv2_s_fold1.json`
- `artifacts/reports/busi_single_comparison_resnet18_fold1.json`
- `artifacts/reports/busi_single_comparison_mobilenetv3_small_100_fold1.json`
- `artifacts/reports/busi_single_swin_tiny_patch4_window7_224_fold1.json`
- `artifacts/reports/busi_single_baseline_resnet18_classifier_fold1.json`
- `artifacts/reports/busi_single_basic_cnn_fold1.json`
- `artifacts/reports/busi_single_convnext_tiny_fold1.json`
- `artifacts/reports/busi_single_comparison_vgg16_fold1.json`
