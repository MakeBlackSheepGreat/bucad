<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Swin-Tiny Five-Fold Training and External Evaluation

Date: 2026-04-25

## Experiment Objective

This round of experiments completes the five-fold training of `swin_tiny_patch4_window7_224` and evaluates its value as a standalone model and subsequent ensemble candidate. Training still strictly uses the BUSBRA training/internal validation split; BUSI is only used for external evaluation and operating point analysis, not for training.

## Training Configuration

| Item | Setting |
| --- | --- |
| Backbone | `swin_tiny_patch4_window7_224` |
| Pre-training | timm ImageNet pre-trained |
| Input | 224 |
| Interpolation & Cropping | bicubic, `crop_pct=0.90` |
| Normalization | mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Ultrasound Preprocessing | CLAHE |
| Augmentation | Horizontal flip |
| Optimizer | AdamW |
| Learning Rate | `5e-5` |
| Weight Decay | `0.05` |
| Class Weights | balanced |
| Schedule | warmup + cosine |
| Checkpoint Selection | Best validation AUC |
| Folds | 5 |

## BUSBRA Five-Fold Internal Validation

| Fold | Best Epoch | AUC | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 30 | 0.9053 | 0.7541 | 0.8933 | 0.8480 | 0.7731 | 0.7635 | TN 226 / FP 27 / FN 30 / TP 92 |
| 2 | 29 | 0.9291 | 0.7851 | 0.9134 | 0.8720 | 0.8120 | 0.7983 | TN 232 / FP 22 / FN 26 / TP 95 |
| 3 | 11 | 0.9300 | 0.7934 | 0.8622 | 0.8400 | 0.7328 | 0.7619 | TN 219 / FP 35 / FN 25 / TP 96 |
| 4 | 16 | 0.9053 | 0.7377 | 0.8854 | 0.8373 | 0.7563 | 0.7469 | TN 224 / FP 29 / FN 32 / TP 90 |
| 5 | 14 | 0.8998 | 0.7273 | 0.8819 | 0.8320 | 0.7458 | 0.7364 | TN 224 / FP 30 / FN 33 / TP 88 |
| Mean | - | 0.9139 | 0.7595 | 0.8872 | 0.8459 | - | - | - |

Internal validation appears stable, with a mean AUC of 0.9139; however, the mean Sensitivity is only 0.7595, indicating that Swin-Tiny remains conservative on BUSBRA internal validation, with unremarkable recall for malignant samples.

## BUSI External Evaluation

| Config | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Swin-Tiny fold1 | 0.8721 | 0.50 | 0.7048 | 0.8902 | 0.8300 | 0.7551 | 0.7291 | TN 389 / FP 48 / FN 62 / TP 148 |
| Swin-Tiny 5-fold + hflip TTA | 0.8867 | 0.50 | 0.7476 | 0.8650 | 0.8269 | 0.7269 | 0.7371 | TN 378 / FP 59 / FN 53 / TP 157 |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.8971 | 0.50 | 0.7762 | 0.8833 | 0.8485 | 0.7617 | 0.7689 | TN 386 / FP 51 / FN 47 / TP 163 |

## BUSI Optimal Youden Operating Point

| Config | AUC | Optimal Threshold | Sensitivity | Specificity | Accuracy | Youden J | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Swin-Tiny fold1 | 0.8721 | 0.18 | 0.7524 | 0.8696 | 0.8315 | 0.6219 | 0.7349 | 0.7435 | TN 380 / FP 57 / FN 52 / TP 158 |
| Swin-Tiny 5-fold + hflip TTA | 0.8867 | 0.67 | 0.7190 | 0.9199 | 0.8547 | 0.6390 | 0.8118 | 0.7626 | TN 402 / FP 35 / FN 59 / TP 151 |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.8971 | 0.50 | 0.7762 | 0.8833 | 0.8485 | 0.6595 | 0.7617 | 0.7689 | TN 386 / FP 51 / FN 47 / TP 163 |

## Comparison with Current Candidate Models

| Model/System | AUC | Recommended Threshold | Sensitivity | Precision | F1-Score | Specificity | Accuracy | Youden J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.8971 | 0.50 | 0.7762 | - | - | 0.8833 | 0.8485 | 0.6595 |
| ConvNeXt-Tiny 5-fold + crop_sweep TTA | 0.9054 | 0.46 | 0.8048 | - | - | 0.8719 | 0.8501 | 0.6766 |
| EfficientNetV2-S + DenseNet121 + ConvNeXt-Tiny Three-Model Ensemble | 0.9138 | 0.32 | 0.8095 | - | - | 0.8627 | 0.8454 | 0.6722 |

## Conclusion

- Swin-Tiny five-fold shows a clear improvement over fold1: AUC increased from 0.8721 to 0.8867, and further to 0.8971 with multi-crop TTA.
- However, Swin-Tiny still falls short of ConvNeXt-Tiny: AUC gap is approximately 0.0083, Youden J gap is approximately 0.0171.
- Swin-Tiny is also significantly lower than the current three-model ensemble: AUC gap is approximately 0.0167.
- Currently not recommended to use Swin-Tiny as the primary model to replace ConvNeXt-Tiny, nor to prioritize significant time investment in further fine-tuning Swin-Tiny.
- If Swin-Tiny is to be used subsequently, a reasonable positioning would be as a "low-weight supplementary member," only added after weight search proves it can improve the three-model ensemble.

## Artifacts

- Training config: `configs/classifier/swin_tiny_patch4_window7_224_timm_recipe.yml`
- Five-fold inference config: `configs/inference/swin_tiny_timm_recipe_5fold.yml`
- Multi-crop TTA config: `configs/inference/swin_tiny_timm_recipe_5fold_tta_crop_sweep.yml`
- BUSI five-fold evaluation: `artifacts/reports/busi_swin_tiny_timm_recipe_5fold.json`
- BUSI multi-crop evaluation: `artifacts/reports/busi_swin_tiny_timm_recipe_5fold_tta_crop_sweep.json`
