# Ensemble Weight Fine Search

Date: 2026-04-24

## Search Setup

- EfficientNetV2-S: five folds, each fold weight `1.0`.
- DenseNet121: five folds, each fold searched from `0.00` to `1.50` with step `0.01`.
- Threshold sweep step: `0.01`.
- BUSI is used for external evaluation only, not training.

## Best Results

| Objective | DenseNet Weight | AUC | Threshold | Sensitivity | Specificity | Accuracy | Youden J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best AUC | 0.63 | 0.9052 | 0.27 | 0.8095 | 0.8719 | 0.8516 | 0.6814 |
| Best Youden | 1.04 | 0.9051 | 0.23 | 0.8238 | 0.8627 | 0.8501 | 0.6865 |
| Best accuracy at Youden point | 0.43 | 0.9048 | 0.29 | 0.8000 | 0.8810 | 0.8547 | 0.6810 |
| Best high-sensitivity point | 0.00 | 0.8997 | 0.24 | 0.8524 | 0.8055 | 0.8207 | 0.6579 |

## Top AUC Weights

| Rank | DenseNet Weight | AUC | Best Threshold | Sensitivity | Specificity | Accuracy | Youden J |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.63 | 0.9052 | 0.27 | 0.8095 | 0.8719 | 0.8516 | 0.6814 |
| 2 | 0.68 | 0.9052 | 0.26 | 0.8143 | 0.8604 | 0.8454 | 0.6747 |
| 3 | 0.71 | 0.9052 | 0.26 | 0.8143 | 0.8627 | 0.8470 | 0.6770 |
| 4 | 0.85 | 0.9052 | 0.25 | 0.8095 | 0.8696 | 0.8501 | 0.6791 |
| 5 | 0.88 | 0.9052 | 0.25 | 0.8048 | 0.8696 | 0.8485 | 0.6743 |
| 6 | 0.89 | 0.9052 | 0.24 | 0.8143 | 0.8604 | 0.8454 | 0.6747 |
| 7 | 0.70 | 0.9052 | 0.26 | 0.8143 | 0.8604 | 0.8454 | 0.6747 |
| 8 | 0.84 | 0.9052 | 0.25 | 0.8143 | 0.8696 | 0.8516 | 0.6839 |
| 9 | 0.62 | 0.9052 | 0.26 | 0.8143 | 0.8650 | 0.8485 | 0.6793 |
| 10 | 0.90 | 0.9052 | 0.24 | 0.8143 | 0.8627 | 0.8470 | 0.6770 |

## Decision

- Best AUC is achieved around DenseNet weight `0.63` with AUC `0.9052`.
- Best Youden operating point is DenseNet weight `1.04` at threshold `0.23`.
- Use the selected objective depending on whether the demo prioritizes AUC ranking or balanced operating-point metrics.
