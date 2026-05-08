<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# ROI Precision / F1 Optimization Study

Date: 2026-04-26

## Objective

- Based on the current dual-model mainline `ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF LCC`, investigate whether the ROI direction can still improve Precision and F1-Score.
- The current mainline issue is not excessive missed diagnoses, but rather an abundance of FP: `TN 357 / FP 80 / FN 31 / TP 179`.
- This round only retains bypass candidates that provide clear benefits, without directly overwriting `configs/inference/demo.yml`.

## Current Mainline

| Approach | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current Dual-Model ROI OOF LCC | 0.9208 | 0.550 | 0.8524 | 0.8169 | 0.8284 | 0.6911 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |

## Tried Directions

| Direction | Representative Config | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Threshold tuning only | Current probability, threshold `0.560` | 0.9208 | 0.560 | 0.8429 | 0.8284 | 0.8331 | 0.7024 | 0.7662 | Abandoned: improvement too small |
| Remove LCC | `demo_roi_oof.yml` | 0.9196 | 0.550 | 0.8429 | 0.8330 | 0.8362 | 0.7080 | 0.7696 | Abandoned: AUC and Sensitivity regression |
| OOF F1 area gate | ROI area `<0.25` falls back to full | 0.9092 | 0.520 | 0.8524 | 0.8284 | 0.8362 | 0.7047 | 0.7716 | Abandoned: AUC drop too large |
| Pareto area gate | ROI area `<0.20` or `>0.95` falls back to full | 0.9141 | 0.520 | 0.8619 | 0.8398 | 0.8470 | 0.7211 | 0.7852 | Abandoned: F1 improved but AUC dropped |
| F1 candidate area gate | ROI area `<0.08` or `>0.75` falls back to full | 0.9256 | 0.510 | 0.8667 | 0.8467 | 0.8532 | 0.7309 | 0.7930 | Retained: all metrics superior to current mainline |

## Retained Candidate

Retained config:

- `configs/inference/demo_roi_area_gate_f1_candidate.yml`

Core logic:

- Segmenter generates ROI mask;
- Uses `mask_threshold=0.40`, `margin_ratio=0.35`, and the largest connected component;
- Calculates the expanded ROI area ratio;
- If ROI area is `<0.08` or `>0.75`, the ROI quality is considered unstable and falls back to full-image probability;
- Otherwise continues using the original ROI OOF stacker;
- Default decision threshold changed to `0.510`.

## Benefits

| Metric | Current Mainline | Area Gate Candidate | Change |
| --- | ---: | ---: | ---: |
| AUC | 0.9208 | 0.9256 | +0.0048 |
| Sensitivity | 0.8524 | 0.8667 | +0.0143 |
| Specificity | 0.8169 | 0.8467 | +0.0297 |
| Accuracy | 0.8284 | 0.8532 | +0.0247 |
| Precision | 0.6911 | 0.7309 | +0.0398 |
| F1-Score | 0.7633 | 0.7930 | +0.0297 |
| FP | 80 | 67 | -13 |
| FN | 31 | 28 | -3 |

## Assessment

- ROI area gating is the most effective ROI direction in this round: it neither continues to expand the ROI nor further lowers the threshold, but instead falls back to full-image when ROI quality is abnormal, reducing benign false positives caused by unreliable ROIs.
- The current candidate simultaneously improves AUC, Sensitivity, Specificity, Accuracy, Precision, and F1-Score, making it worth retaining as the next mainline candidate.
- Still not recommended to immediately overwrite `demo.yml`: the next step should be to re-check the area thresholds on BUSBRA OOF or the team's fixed validation protocol before deciding whether to formally merge into the default demo.
