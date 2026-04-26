# Resolution And Augmentation Experiment

Date: 2026-04-24

## Goal

Evaluate whether the classifier should move from the frozen `224` pipeline to a
higher-resolution `256` pipeline and whether additional ultrasound-safe
augmentation improves fold-1 validation behavior.

## Configurations

| Experiment | Config | Image Size | Extra Augmentation |
| --- | --- | ---: | --- |
| Baseline | `configs/classifier/efficientnetv2_s.yml` | 224 | horizontal flip |
| Resolution only | `configs/classifier/efficientnetv2_s_256.yml` | 256 | horizontal flip |
| Resolution + mild aug | `configs/classifier/efficientnetv2_s_256_aug.yml` | 256 | horizontal flip, rotation, brightness, contrast, scale crop |
| Resolution 320 | `configs/classifier/efficientnetv2_s_320.yml` | 320 | horizontal flip |

## Fold-1 Results

| Experiment | AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 baseline | 0.9248 | 0.6148 | - | - | 0.9565 | 0.8453 |
| 256 resolution only | 0.8994 | 0.7213 | - | - | 0.9328 | 0.8640 |
| 256 mild augmentation | 0.8811 | 0.7541 | - | - | 0.8379 | 0.8107 |
| 320 resolution only | 0.8903 | 0.7541 | - | - | 0.8735 | 0.8347 |

## Decision

- Do not replace the frozen runtime model with either `256` experiment yet.
- The `256` resolution-only run improves fold-1 sensitivity and accuracy, but
  lowers AUC substantially compared with the current `224` baseline.
- The additional mild augmentation further increases sensitivity but reduces AUC,
  specificity, and accuracy, suggesting that the current augmentation strength is
  not a good final-model direction.
- The `320` resolution-only run did not recover the AUC loss seen at `256`; it
  improves sensitivity but reduces AUC, specificity, and accuracy versus the
  frozen `224` baseline.
- Keep the new configs and augmentation code for reproducible follow-up
  experiments, but keep `configs/inference/demo.yml` on the existing five-fold
  EfficientNetV2-S runtime ensemble.

## Follow-Up Options

- Try `256` with lower learning rate such as `0.0001`.
- Try `256` with early stopping or best-validation checkpoint saving.
- Try mixed ensembling between the current `224` fold models and selected `256`
  models only if BUSI external evaluation confirms a gain.
