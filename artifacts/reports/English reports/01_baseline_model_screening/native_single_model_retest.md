<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Native/Recommended Single Model Retest Report

Date: 2026-04-26

## Description

- This report retests the available single-fold native or recommended protocol models in the project, excluding subsequent low-learning-rate, resolution, sensitivity, seed, ROI, OOF, or ensemble tuning variants.
- Evaluation output uniformly includes AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score.
- `AlexNet` failed in existing training records and has no checkpoint available for evaluation, so it is still not included in the table.
- `ConvNeXt-Small` is a supplementary model from a subsequent upgrade experiment, not part of the original 4-model mainline screening, but is listed for reference.

## Default Threshold 0.50 Retest Results

| Rank | Model | Threshold | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Confusion | Protocol | Notes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | ConvNeXt-Small timm recipe fold1 | 0.50 | 0.8947 | 0.8284 | 0.7667 | 0.7220 | 0.8581 | 0.7436 | TN 375 / FP 62 / FN 49 / TP 161 | 224 + CLAHE + timm mean/std + bicubic + crop 0.95 + hflip TTA | Subsequent upgrade verification, not part of default mainline |
| 2 | ConvNeXt-Tiny timm recipe fold1 | 0.50 | 0.8943 | 0.8423 | 0.7762 | 0.7477 | 0.8741 | 0.7617 | TN 382 / FP 55 / FN 47 / TP 163 | 224 + CLAHE + timm mean/std + bicubic + crop 0.95 + hflip TTA | ConvNeXt recommended recipe |
| 3 | DenseNet121 fold1 | 0.50 | 0.8766 | 0.8083 | 0.4571 | 0.9057 | 0.9771 | 0.6076 | TN 427 / FP 10 / FN 114 / TP 96 | 224 + CLAHE + hflip TTA | Dense connectivity candidate |
| 4 | Swin-Tiny timm recipe fold1 | 0.50 | 0.8729 | 0.8300 | 0.7048 | 0.7551 | 0.8902 | 0.7291 | TN 389 / FP 48 / FN 62 / TP 148 | 224 + CLAHE + timm mean/std + bicubic + crop 0.90 + hflip TTA | Swin recommended recipe |
| 5 | EfficientNetV2-S fold1 | 0.50 | 0.8609 | 0.7465 | 0.8333 | 0.5757 | 0.7048 | 0.6809 | TN 308 / FP 129 / FN 35 / TP 175 | 224 + CLAHE + hflip TTA | EfficientNetV2 candidate |
| 6 | EfficientNetV2-S main fold1 | 0.50 | 0.8483 | 0.8099 | 0.6714 | 0.7231 | 0.8764 | 0.6963 | TN 383 / FP 54 / FN 69 / TP 141 | 224 + CLAHE + hflip TTA | Fold1 of the subsequent 5-fold mainline EfficientNet branch |
| 7 | ResNet18 fold1 | 0.50 | 0.8480 | 0.7991 | 0.7714 | 0.6639 | 0.8124 | 0.7137 | TN 355 / FP 82 / FN 48 / TP 162 | 224 + CLAHE + hflip TTA | ResNet baseline |
| 8 | MobileNetV3-Small fold1 | 0.50 | 0.8431 | 0.7543 | 0.7143 | 0.6024 | 0.7735 | 0.6536 | TN 338 / FP 99 / FN 60 / TP 150 | 224 + CLAHE + hflip TTA | MobileNet lightweight baseline |
| 9 | Swin-Tiny initial fold1 | 0.50 | 0.8242 | 0.6754 | 0.8714 | 0.5000 | 0.5812 | 0.6354 | TN 254 / FP 183 / FN 27 / TP 183 | 224 + CLAHE + hflip TTA, no timm recipe | Early non-timm-aware Swin |
| 10 | Early ResNet18 classifier fold1 | 0.50 | 0.7543 | 0.7450 | 0.6429 | 0.6000 | 0.7941 | 0.6207 | TN 347 / FP 90 / FN 75 / TP 135 | 224 + identity TTA | Early ResNet18 baseline |
| 11 | Basic CNN fold1 | 0.50 | 0.7327 | 0.6754 | 0.0429 | 0.5000 | 0.9794 | 0.0789 | TN 428 / FP 9 / FN 201 / TP 9 | 224 + CLAHE + hflip TTA | Non-pretrained lightweight CNN baseline |
| 12 | ConvNeXt-Tiny initial fold1 | 0.50 | 0.5996 | 0.6754 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | TN 437 / FP 0 / FN 210 / TP 0 | 224 + CLAHE + hflip TTA, no timm recipe | Early non-timm-aware ConvNeXt |
| 13 | VGG16 fold1 | 0.50 | 0.5000 | 0.6754 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | TN 437 / FP 0 / FN 210 / TP 0 | 224 + CLAHE + hflip TTA | VGG baseline |

## Youden-Optimal Operating Points

| Rank | Model | Threshold | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Youden J | Confusion |
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

## Origin of the Four Candidate Models

- During the initial screening phase, DenseNet121, EfficientNetV2-S, ConvNeXt-Tiny, and Swin-Tiny showed the best single-fold performance, structural complementarity, or subsequent optimization potential, and were therefore selected as the main candidates for subsequent BUCAD 5-fold training and hybrid ensemble.
- ConvNeXt-Tiny significantly improved from its early failure state after introducing timm-aware preprocessing, and became the core candidate for the subsequent main model.
- EfficientNetV2-S and DenseNet121 performed stably among the CNN family, providing better overall performance and higher Precision/Specificity references, respectively.
- Swin-Tiny's single-fold performance was inferior to ConvNeXt-Tiny, but as a Transformer-style candidate it offers complementarity, and thus entered subsequent 5-fold and hybrid ensemble testing.
- The subsequent formal demo did not directly adopt a full 4-model ensemble, but further converged to `ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate` based on 5-fold, TTA, OOF, ROI, and deployment complexity considerations.

## Output Files

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
