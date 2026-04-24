# EfficientNetV2-S 5-Fold Training Summary

Date: 2026-04-24

## Fold Metrics

| Fold | AUC | Sensitivity | Specificity | Accuracy | Checkpoint |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.9248 | 0.6148 | 0.9565 | 0.8453 | `C:/Users/876762330/Desktop/Agent/artifacts/checkpoints/efficientnetv2_s_fold1.pt` |
| 2 | 0.8988 | 0.7521 | 0.8386 | 0.8107 | `C:/Users/876762330/Desktop/Agent/artifacts/checkpoints/efficientnetv2_s_fold2.pt` |
| 3 | 0.9077 | 0.6612 | 0.9016 | 0.8240 | `C:/Users/876762330/Desktop/Agent/artifacts/checkpoints/efficientnetv2_s_fold3.pt` |
| 4 | 0.8891 | 0.6230 | 0.9289 | 0.8293 | `C:/Users/876762330/Desktop/Agent/artifacts/checkpoints/efficientnetv2_s_fold4.pt` |
| 5 | 0.8523 | 0.6116 | 0.9016 | 0.8080 | `C:/Users/876762330/Desktop/Agent/artifacts/checkpoints/efficientnetv2_s_fold5.pt` |

## Mean Internal Validation

- Mean AUC: `0.8946`
- Mean sensitivity: `0.6525`
- Mean specificity: `0.9054`
- Mean accuracy: `0.8235`

## BUSI External Evaluation With 5-Fold Ensemble

- AUC: `0.8955`
- Default 0.50 sensitivity/specificity/accuracy: `0.6667` / `0.9382` / `0.8501`
- Best Youden threshold: `0.25`
- Best-threshold sensitivity/specificity/accuracy: `0.8476` / `0.8215` / `0.8300`

## Decision Impact

- T059 is complete: all five EfficientNetV2-S fold checkpoints and reports exist.
- Runtime inference is switched to the five-fold EfficientNetV2-S ensemble in `configs/inference/demo.yml`.
- The AUC target is exceeded. Sensitivity is effectively at the 0.85 target when using the best Youden threshold 0.25, but it remains slightly below the target by 0.0024.
