<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# ROI Soft Gate + Multi-Scale Cropping OOF Experiment

Date: 2026-04-26

## Experiment Boundary

- Candidate parameters are selected using BUSBRA OOF only.
- BUSI is used only for a single post-fixation review after the candidate is fixed; it does not participate in the search.
- The current mainline uses ROI area hard gating; this experiment attempts to replace hard fallback with continuous weighting.

## OOF Selected Scheme

- margins: `[0.2, 0.35]`
- margin_weights: `[0.35, 0.65]`
- min_area_ratio: `0.15`
- max_area_ratio: `1.01`
- ramp_width: `0.03`
- max_weight: `1.0`
- mean_gate_weight: `0.8097`
- OOF threshold: `0.550`

## Full Metric Comparison

| Scheme | Samples | Positive | Negative | AUC | Threshold | Accuracy | Sensitivity | Recall | Specificity | Precision | NPV | F1-Score | Balanced Acc | Youden J | FPR | FNR | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OOF current area gate | 1875 | 607 | 1268 | 0.9166 | 0.510 | 0.8416 | 0.8402 | 0.8402 | 0.8423 | 0.7183 | 0.9167 | 0.7745 | 0.8412 | 0.6825 | 0.1577 | 0.1598 | TN 1068 / FP 200 / FN 97 / TP 510 |
| OOF soft gate multiscale | 1875 | 607 | 1268 | 0.9221 | 0.550 | 0.8597 | 0.8402 | 0.8402 | 0.8691 | 0.7544 | 0.9191 | 0.7950 | 0.8546 | 0.7093 | 0.1309 | 0.1598 | TN 1102 / FP 166 / FN 97 / TP 510 |
| External current area gate | 647 | 210 | 437 | 0.9256 | 0.510 | 0.8532 | 0.8667 | 0.8667 | 0.8467 | 0.7309 | 0.9296 | 0.7930 | 0.8567 | 0.7133 | 0.1533 | 0.1333 | TN 370 / FP 67 / FN 28 / TP 182 |
| External soft gate multiscale | 647 | 210 | 437 | 0.9189 | 0.550 | 0.8393 | 0.8429 | 0.8429 | 0.8375 | 0.7137 | 0.9173 | 0.7729 | 0.8402 | 0.6804 | 0.1625 | 0.1571 | TN 366 / FP 71 / FN 33 / TP 177 |

## OOF Top Candidates

| Rank | margins | weights | min | max | ramp | max_weight | AUC | Threshold | Sensitivity | Specificity | Precision | F1 | FPR | FNR |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `[0.2, 0.35]` | `[0.35, 0.65]` | 0.15 | 1.01 | 0.03 | 1.00 | 0.9221 | 0.550 | 0.8402 | 0.8691 | 0.7544 | 0.7950 | 0.1309 | 0.1598 |
| 2 | `[0.35, 0.5]` | `[0.65, 0.35]` | 0.15 | 0.95 | 0.03 | 1.00 | 0.9213 | 0.530 | 0.8402 | 0.8675 | 0.7522 | 0.7938 | 0.1325 | 0.1598 |
| 3 | `[0.5]` | `[1.0]` | 0.15 | 1.01 | 0.03 | 1.00 | 0.9217 | 0.520 | 0.8418 | 0.8659 | 0.7504 | 0.7935 | 0.1341 | 0.1582 |
| 4 | `[0.2, 0.35, 0.5]` | `[0.2, 0.3, 0.5]` | 0.15 | 1.01 | 0.03 | 1.00 | 0.9222 | 0.530 | 0.8435 | 0.8644 | 0.7485 | 0.7932 | 0.1356 | 0.1565 |
| 5 | `[0.5]` | `[1.0]` | 0.15 | 1.01 | 0.03 | 0.80 | 0.9217 | 0.470 | 0.8484 | 0.8604 | 0.7442 | 0.7929 | 0.1396 | 0.1516 |
| 6 | `[0.35, 0.5]` | `[0.65, 0.35]` | 0.15 | 1.01 | 0.03 | 1.00 | 0.9220 | 0.530 | 0.8451 | 0.8628 | 0.7467 | 0.7929 | 0.1372 | 0.1549 |
| 7 | `[0.35, 0.5]` | `[0.35, 0.65]` | 0.15 | 1.01 | 0.03 | 1.00 | 0.9221 | 0.530 | 0.8418 | 0.8651 | 0.7493 | 0.7929 | 0.1349 | 0.1582 |
| 8 | `[0.2, 0.5]` | `[0.35, 0.65]` | 0.15 | 1.01 | 0.03 | 1.00 | 0.9220 | 0.530 | 0.8418 | 0.8651 | 0.7493 | 0.7929 | 0.1349 | 0.1582 |
| 9 | `[0.2, 0.35, 0.5]` | `[0.5, 0.3, 0.2]` | 0.15 | 1.01 | 0.08 | 1.00 | 0.9219 | 0.520 | 0.8418 | 0.8651 | 0.7493 | 0.7929 | 0.1349 | 0.1582 |
| 10 | `[0.2, 0.5]` | `[0.35, 0.65]` | 0.12 | 1.01 | 0.05 | 1.00 | 0.9225 | 0.540 | 0.8402 | 0.8659 | 0.7500 | 0.7925 | 0.1341 | 0.1598 |

## Conclusion

- The post-fixation review did not exceed the current mainline; this direction is recommended for abandonment.
