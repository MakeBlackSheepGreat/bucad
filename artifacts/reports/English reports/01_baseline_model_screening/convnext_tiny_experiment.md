# ConvNeXt-Tiny Experiment

Date: 2026-04-25

## Goal

Evaluate whether `convnext_tiny` is worth adding to the breast ultrasound
classifier candidate pool. This is a fold-1 single-model experiment, so it is
used for backbone screening only. It should not be compared directly against
the final optimized mixed ensemble.

## Setup

| Item | Setting |
| --- | --- |
| Model | `convnext_tiny` |
| Training data | BUSBRA only |
| Fold | 1 |
| Epochs | 20 |
| Input size | 224 |
| Preprocessing | CLAHE |
| Augmentation | Horizontal flip |
| Learning rate | 0.0003 |
| BUSI role | External evaluation only; not used for training |

## Recipe Limitation

This was a quick project-recipe screening run, not a full ConvNeXt-specific
optimization. The run reused the current classifier pipeline: square resize,
CLAHE, horizontal flip, AdamW, fixed learning rate, no scheduler/warmup, and
last-checkpoint saving. It did not apply timm's full pretrained transform and
training recipe for ConvNeXt, such as ImageNet mean/std normalization, crop
policy, lower fine-tuning learning rate, warmup/scheduler, or layer-wise
learning-rate decay.

Therefore this result shows that the current quick ConvNeXt recipe failed. It
does not prove ConvNeXt is fundamentally unsuitable.

## Fold-1 Single-Model Comparison

| Model | AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNetV2-S | 0.8937 | 0.7049 | - | - | 0.8775 | 0.8213 |
| DenseNet121 | 0.8867 | 0.7213 | - | - | 0.8775 | 0.8267 |
| ResNet18 | 0.8756 | 0.7131 | - | - | 0.8577 | 0.8107 |
| MobileNetV3-Small | 0.8704 | 0.6557 | - | - | 0.9170 | 0.8320 |
| Swin-Tiny | 0.7950 | 0.8852 | - | - | 0.5020 | 0.6267 |
| Basic CNN | 0.6427 | 0.0000 | - | - | 0.9921 | 0.6693 |
| VGG16 | 0.5000 | 0.0000 | - | - | 1.0000 | 0.6747 |
| ConvNeXt-Tiny | 0.4519 | 0.0000 | - | - | 1.0000 | 0.6747 |

Under the current training recipe, ConvNeXt-Tiny does not learn a useful
malignant decision boundary. The final checkpoint predicts all validation
samples as benign at threshold 0.50. The best observed epoch was epoch 3, but
its AUC was only 0.5455 and still predicted no malignant samples at the default
threshold.

## BUSI External Sanity Check

| Operating Point | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default threshold | 0.6012 | 0.50 | 0.0000 | 1.0000 | 0.6754 | 0.0000 | 0.0000 | TN 437 / FP 0 / FN 210 / TP 0 |
| Best sweep point | 0.6012 | 0.10 | 1.0000 | 0.0000 | 0.3246 | 0.3246 | 0.4901 | TN 0 / FP 437 / FN 0 / TP 210 |

The BUSI probabilities are poorly calibrated for normal threshold use. At
threshold 0.50 the model predicts all samples as benign; at the lowest swept
threshold it predicts all samples as malignant. This is not useful for the
current diagnostic workflow.

## Decision

- Do not add this ConvNeXt-Tiny checkpoint to the ensemble.
- Do not run five folds for this exact configuration.
- This result does not prove ConvNeXt can never work; it shows that
  `convnext_tiny` with the current 224 / CLAHE / horizontal-flip / lr=0.0003
  recipe is not competitive.
- If ConvNeXt is retried, use a separate optimized recipe: lower learning rate,
  best-epoch checkpoint selection, class balancing, and probability calibration.

## Artifacts

- Config: `configs/classifier/convnext_tiny.yml`
- Training report: `artifacts/reports/train_cls_convnext_tiny_fold1.json`
- Checkpoint: `artifacts/checkpoints/convnext_tiny_fold1.pt`
- BUSI report: `artifacts/reports/busi_convnext_tiny_fold1.json`
- BUSI threshold analysis: `artifacts/reports/threshold_analysis_convnext_tiny_fold1.md`
