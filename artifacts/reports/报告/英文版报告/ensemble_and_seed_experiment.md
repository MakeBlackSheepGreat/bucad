# Ensemble And Seed Experiment

Date: 2026-04-24

## Goal

Test two higher-priority improvement directions:

1. Heterogeneous model ensembling.
2. Low-learning-rate retraining with a different seed.

BUSI is used only for external evaluation, not training.

## Mixed Ensemble Results On BUSI

| Ensemble | AUC | Sensitivity @0.50 | Precision | F1-Score | Specificity @0.50 | Accuracy @0.50 | Best Threshold | Best Sensitivity | Best Specificity | Best Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNetV2-S 5-fold + TTA | 0.8997 | 0.6619 | - | - | 0.9382 | 0.8485 | 0.33 | 0.8000 | 0.8764 | 0.8516 |
| EfficientNetV2-S 5-fold + DenseNet121 fold1 weight 0.25 | 0.9011 | 0.6476 | - | - | 0.9428 | 0.8470 | 0.32 | 0.8000 | 0.8764 | 0.8516 |
| EfficientNetV2-S 5-fold + DenseNet121 fold1 weight 0.50 | 0.9021 | 0.6476 | - | - | 0.9428 | 0.8470 | 0.30 | 0.8095 | 0.8719 | 0.8516 |
| EfficientNetV2-S 5-fold + DenseNet121 5-fold weight 0.25 | 0.9037 | 0.6286 | - | - | 0.9474 | 0.8439 | 0.29 | 0.8095 | 0.8719 | 0.8516 |
| EfficientNetV2-S 5-fold + DenseNet121 5-fold weight 0.50 | 0.9050 | 0.5905 | - | - | 0.9519 | 0.8346 | 0.28 | 0.8095 | 0.8696 | 0.8501 |
| EfficientNetV2-S 5-fold + DenseNet121 5-fold weight 0.63 | 0.9052 | 0.5810 | - | - | 0.9542 | 0.8331 | 0.27 | 0.8095 | 0.8719 | 0.8516 |
| EfficientNetV2-S 5-fold + DenseNet121 5-fold weight 1.00 | 0.9051 | 0.5714 | - | - | 0.9611 | 0.8346 | 0.23 | 0.8190 | 0.8581 | 0.8454 |
| EfficientNetV2-S fold1 + DenseNet121 fold1 | 0.8767 | 0.5524 | - | - | 0.9497 | 0.8207 | 0.15 | - | - | - |
| EfficientNetV2-S fold1 + ResNet18 fold1 | 0.8612 | 0.6857 | - | - | 0.8879 | 0.8223 | 0.29 | - | - | - |
| EfficientNetV2-S fold1 + MobileNetV3 fold1 | 0.8629 | 0.7095 | - | - | 0.8764 | 0.8223 | 0.38 | - | - | - |

## Low-Learning-Rate Seed Result

| Experiment | Config | Best Epoch | AUC | Sensitivity @0.50 | Precision | F1-Score | Specificity @0.50 | Accuracy @0.50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline EfficientNetV2-S fold1 | `configs/classifier/efficientnetv2_s.yml` | - | 0.9248 | 0.6148 | - | - | 0.9565 | 0.8453 |
| LR 1e-4 seed 123 fold1 | `configs/classifier/efficientnetv2_s_lr1e4_seed123.yml` | 27 | 0.8503 | 0.6639 | - | - | 0.8656 | 0.8000 |

## Decision

- The best immediate improvement is heterogeneous ensembling. A complete
  DenseNet121 five-fold add-on improves BUSI AUC from `0.8997` to `0.9052` at
  the selected balanced runtime point.
- Single-fold heterogeneous ensembles are weaker than the current five-fold
  EfficientNetV2-S ensemble, so mixing should be done as a small add-on to the
  existing ensemble rather than replacing it.
- The first low-learning-rate seed run did not help; do not continue large
  low-learning-rate seed sweeps unless there is time for multiple seeds and a
  stronger checkpoint-selection plan.
- Recommended next step: keep the mixed ensemble as the runtime candidate and
  continue additional seed or architecture experiments only if they improve on
  the `0.9052` BUSI AUC baseline.
