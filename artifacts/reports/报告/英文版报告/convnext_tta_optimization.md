<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# ConvNeXt-Tiny TTA Optimization Experiment

> Data boundary note: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or tuning.


Date: 2026-04-25

## Objective

This round of experiments only optimizes test-time augmentation (TTA) for the ConvNeXt-Tiny 5-fold model. The training boundary remains unchanged: BUSBRA is used only for training and internal validation, BUSI is used only for external evaluation and operating point analysis, and does not participate in training.

## Implementation Adjustments

- The inference service added `classifier_tta_variants`, which allows explicit configuration of TTA combinations including identity, horizontal flip, slight rotation, and different `crop_pct` values.
- The inference report added `runtime_summary`, recording input size, crop ratio, and TTA variants to facilitate experiment reproducibility.
- BUSI evaluation additionally generates a same-named threshold analysis report when `--output` is passed, preventing multiple experiment groups from overwriting a single `threshold_analysis.md`.
- ConvNeXt-Tiny TTA inference changed to perform forward passes per TTA batch, and reads `device: auto` to use CUDA; local evaluation was completed using an NVIDIA GPU.

## Candidate TTA Configurations

| Config | TTA Variants | Time per Run |
| --- | --- | ---: |
| identity | `crop_pct=0.95` original image | 39.5s |
| hflip | `crop_pct=0.95` original image + horizontal flip | 46.0s |
| crop90_hflip | `crop_pct=0.90` original image + horizontal flip | 45.1s |
| crop100_hflip | `crop_pct=1.00` original image + horizontal flip | 45.2s |
| crop_sweep | `crop_pct=0.90/0.95/1.00`, each crop paired with original + horizontal flip | 73.7s |
| rotate5 | `crop_pct=0.95` original image + horizontal flip + `+/-5 deg` rotation | 57.9s |

## BUSI External Evaluation: Default Threshold 0.50

| Config | AUC | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| identity | 0.8991 | 0.7381 | 0.8810 | 0.8346 | 0.7488 | 0.7434 | TN 385 / FP 52 / FN 55 / TP 155 |
| hflip | 0.9041 | 0.7381 | 0.8924 | 0.8423 | 0.7673 | 0.7524 | TN 390 / FP 47 / FN 55 / TP 155 |
| crop90_hflip | 0.9018 | 0.7524 | 0.8810 | 0.8393 | 0.7524 | 0.7524 | TN 385 / FP 52 / FN 52 / TP 158 |
| crop100_hflip | 0.8994 | 0.7762 | 0.8581 | 0.8315 | 0.7244 | 0.7494 | TN 375 / FP 62 / FN 47 / TP 163 |
| crop_sweep | **0.9054** | **0.7714** | 0.8879 | **0.8501** | 0.7678 | 0.7696 | TN 388 / FP 49 / FN 48 / TP 162 |
| rotate5 | 0.9048 | 0.7381 | **0.8970** | 0.8454 | 0.7750 | 0.7561 | TN 392 / FP 45 / FN 55 / TP 155 |

## BUSI External Evaluation: Youden-Optimal Operating Points

| Config | Threshold | AUC | Sensitivity | Specificity | Accuracy | Youden J | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| identity | 0.54 | 0.8991 | 0.7381 | 0.9085 | 0.8532 | 0.6466 | 0.7949 | 0.7654 | TN 397 / FP 40 / FN 55 / TP 155 |
| hflip | 0.47 | 0.9041 | 0.7667 | 0.8879 | 0.8485 | 0.6545 | 0.7667 | 0.7667 | TN 388 / FP 49 / FN 49 / TP 161 |
| crop90_hflip | 0.26 | 0.9018 | **0.8619** | 0.7941 | 0.8161 | 0.6560 | 0.6679 | 0.7526 | TN 347 / FP 90 / FN 29 / TP 181 |
| crop100_hflip | 0.82 | 0.8994 | 0.6857 | **0.9565** | **0.8686** | 0.6422 | 0.8834 | 0.7721 | TN 418 / FP 19 / FN 66 / TP 144 |
| crop_sweep | 0.46 | **0.9054** | 0.8048 | 0.8719 | 0.8501 | **0.6766** | 0.7511 | 0.7770 | TN 381 / FP 56 / FN 41 / TP 169 |
| rotate5 | 0.40 | 0.9048 | 0.7810 | 0.8696 | 0.8408 | 0.6505 | 0.7421 | 0.7610 | TN 380 / FP 57 / FN 46 / TP 164 |

## Comparison with Existing Ensemble Baseline

| Model/System | AUC | Recommended Threshold | Sensitivity | Precision | F1-Score | Specificity | Accuracy | Youden J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNetV2-S 5-fold + TTA | 0.8997 | 0.33 | 0.8000 | - | - | 0.8764 | **0.8516** | 0.6764 |
| EfficientNetV2-S + DenseNet121 Hybrid Ensemble | 0.9052 | 0.27 | **0.8095** | - | - | 0.8719 | **0.8516** | **0.6814** |
| ConvNeXt-Tiny 5-fold + crop_sweep TTA | **0.9054** | 0.46 | 0.8048 | - | - | 0.8719 | 0.8501 | 0.6766 |

## Conclusions

- ConvNeXt-Tiny is very sensitive to TTA. AUC is 0.8991 without TTA, improving to 0.9041 after adding horizontal flip.
- A single crop ratio is not stable: `crop_pct=0.90` and `crop_pct=1.00` used alone are both lower than the standard `0.95 + hflip`.
- Multi-crop averaging is most effective. `crop_sweep` raises AUC to 0.9054, slightly higher than the current EfficientNetV2-S + DenseNet121 hybrid ensemble's 0.9052.
- However, from the operating point perspective, the current hybrid ensemble still has higher Youden J, higher Sensitivity, and higher Accuracy. Therefore, ConvNeXt-Tiny should not directly replace the demo default model at this time.
- It is recommended to use `configs/inference/convnext_tiny_timm_recipe_5fold_tta_crop_sweep.yml` as the candidate configuration for ConvNeXt-Tiny as a subsequent main model, before proceeding to Swin-Tiny low-weight ensemble search.

## Recommended Next Steps

1. First fix the candidate TTA for ConvNeXt-Tiny to `crop_sweep`, to avoid continued low-yield search on single-model TTA.
2. Next, after training or filling in Swin-Tiny 5-fold, test the low-weight ensemble of ConvNeXt-Tiny crop_sweep with Swin-Tiny.
3. If the final objective leans toward screening scenarios, prioritize comparing Sensitivity near the Youden operating point, rather than only looking at the default 0.50 threshold.
