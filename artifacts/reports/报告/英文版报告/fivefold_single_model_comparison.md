<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Four 5-Fold Single Model Performance Comparison

> Data boundary note: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or tuning.


Date: 2026-04-25

## Description

- This report only compares 5-fold results of single model families, and does not include cross-model hybrid ensembles.
- BUSBRA is used for training and internal validation; BUSI is used only for external evaluation and operating point analysis.
- ConvNeXt-Tiny and Swin-Tiny use their respective timm-aware preprocessing; EfficientNetV2-S and DenseNet121 use the original 224 + CLAHE + hflip TTA protocol.

## BUSBRA Internal 5-Fold Mean

| Model | Mean AUC | Mean Sens | Precision | F1-Score | Mean Spec | Mean Acc | Protocol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EfficientNetV2-S 5-fold + hflip TTA | 0.8946 | 0.6525 | - | - | 0.9054 | 0.8235 | 224 + CLAHE + hflip TTA |
| DenseNet121 5-fold + hflip TTA | 0.8749 | 0.6489 | - | - | 0.9148 | 0.8288 | 224 + CLAHE + hflip TTA |
| ConvNeXt-Tiny 5-fold + crop_sweep TTA | 0.9212 | 0.7744 | - | - | 0.8864 | 0.8501 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.9139 | 0.7595 | - | - | 0.8872 | 0.8459 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |

## BUSI External Evaluation: Default Threshold 0.50

| Model | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | Precision | F1-Score | Confusion | Protocol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ConvNeXt-Tiny 5-fold + crop_sweep TTA | 0.9054 | 0.7714 | 0.8879 | 0.8501 | 0.7678 | 0.7696 | TN 388 / FP 49 / FN 48 / TP 162 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |
| EfficientNetV2-S 5-fold + hflip TTA | 0.8997 | 0.6619 | 0.9382 | 0.8485 | 0.8373 | 0.7394 | TN 410 / FP 27 / FN 71 / TP 139 | 224 + CLAHE + hflip TTA |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.8971 | 0.7762 | 0.8833 | 0.8485 | 0.7617 | 0.7689 | TN 386 / FP 51 / FN 47 / TP 163 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |
| DenseNet121 5-fold + hflip TTA | 0.8914 | 0.4952 | 0.9794 | 0.8223 | 0.9204 | 0.6440 | TN 428 / FP 9 / FN 106 / TP 104 | 224 + CLAHE + hflip TTA |

## BUSI External Evaluation: Youden-Optimal Operating Points

| Model | AUC | Threshold | Sens | Spec | Acc | Youden J | Precision | F1-Score | Confusion | Protocol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ConvNeXt-Tiny 5-fold + crop_sweep TTA | 0.9054 | 0.46 | 0.8048 | 0.8719 | 0.8501 | 0.6766 | 0.7511 | 0.7770 | TN 381 / FP 56 / FN 41 / TP 169 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |
| EfficientNetV2-S 5-fold + hflip TTA | 0.8997 | 0.33 | 0.8000 | 0.8764 | 0.8516 | 0.6764 | 0.7568 | 0.7778 | TN 383 / FP 54 / FN 42 / TP 168 | 224 + CLAHE + hflip TTA |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.8971 | 0.50 | 0.7762 | 0.8833 | 0.8485 | 0.6595 | 0.7617 | 0.7689 | TN 386 / FP 51 / FN 47 / TP 163 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |
| DenseNet121 5-fold + hflip TTA | 0.8914 | 0.19 | 0.7952 | 0.8581 | 0.8377 | 0.6534 | 0.7293 | 0.7608 | TN 375 / FP 62 / FN 43 / TP 167 | 224 + CLAHE + hflip TTA |

## Conclusions

- Among 5-fold single models, the highest BUSI AUC belongs to ConvNeXt-Tiny 5-fold + crop_sweep TTA, with AUC `0.9054`.
- The highest Youden J belongs to ConvNeXt-Tiny 5-fold + crop_sweep TTA, with Youden J `0.6766`.
- ConvNeXt-Tiny is the strongest single-model 5-fold candidate; although Swin-Tiny 5-fold improved over fold1, it is still lower than ConvNeXt-Tiny.
