# Sensitivity Tuning Experiment

Date: 2026-04-24

## Goal

Improve the low fold-1 sensitivity of the `224` EfficientNetV2-S classifier
without using BUSI for training.

## Methods

| Experiment | Config | Main Change | Checkpoint Selection |
| --- | --- | --- | --- |
| Baseline | `configs/classifier/efficientnetv2_s.yml` | original 224 training | final epoch |
| Sensitive | `configs/classifier/efficientnetv2_s_sensitive.yml` | malignant class weight 1.35 | threshold 0.24 sensitivity-weighted score |
| Balanced sensitive | `configs/classifier/efficientnetv2_s_sensitive_balanced.yml` | malignant class weight 1.20 | threshold 0.24 score with specificity constraint |

## Fold-1 Validation Results

| Experiment | Best Epoch | AUC | Sensitivity @0.50 | Specificity @0.50 | Accuracy @0.50 | Sensitivity @0.24 | Specificity @0.24 | Accuracy @0.24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 0.9248 | 0.6148 | 0.9565 | 0.8453 | - | - | - |
| Sensitive | 18 | 0.8922 | 0.9016 | 0.7075 | 0.7707 | 0.9262 | 0.6482 | 0.7387 |
| Balanced sensitive | 27 | 0.9170 | 0.7541 | 0.8775 | 0.8373 | 0.8033 | 0.8696 | 0.8480 |

## Decision

- Do not replace the frozen five-fold runtime ensemble based on a single fold.
- The aggressive sensitive config greatly improves sensitivity but sacrifices too
  much specificity and AUC.
- The balanced sensitive config is the best follow-up candidate: it improves
  default-threshold sensitivity from `0.6148` to `0.7541` while keeping AUC close
  to the baseline (`0.9170` vs `0.9248`).
- If more training time is available, run all five folds for
  `configs/classifier/efficientnetv2_s_sensitive_balanced.yml` and evaluate the
  resulting ensemble on BUSI before considering a runtime switch.
