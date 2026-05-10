<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# ROI Area Gate OOF Fixed Protocol

Date: 2026-04-26

## Protocol

- Only BUSBRA OOF predictions are used to select ROI area gate parameters and threshold.
- The objective is to improve Precision and F1-Score without dropping Sensitivity below a set floor.
- The selection process does not read external evaluation results; external results are used only for a single post-fixation review.

## OOF Baseline

| AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.9239 | 0.550 | 0.8534 | 0.8423 | 0.8459 | 0.7214 | 0.7819 | TN 1068 / FP 200 / FN 89 / TP 518 |

## OOF Selected Candidate

| min_area | max_area | fallback_count | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 1.01 | 120 | 0.9233 | 0.560 | 0.8435 | 0.8596 | 0.8544 | 0.7420 | 0.7895 |

## Top OOF Candidates

| Rank | min_area | max_area | accepted | AUC | Threshold | Sensitivity | Precision | F1-Score |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.10 | 1.01 | True | 0.9233 | 0.560 | 0.8435 | 0.7420 | 0.7895 |
| 2 | 0.08 | 1.01 | True | 0.9239 | 0.560 | 0.8435 | 0.7388 | 0.7877 |
| 3 | 0.03 | 1.01 | True | 0.9239 | 0.560 | 0.8451 | 0.7329 | 0.7850 |
| 4 | 0.00 | 1.01 | True | 0.9239 | 0.560 | 0.8451 | 0.7329 | 0.7850 |
| 5 | 0.01 | 1.01 | True | 0.9239 | 0.560 | 0.8451 | 0.7329 | 0.7850 |
| 6 | 0.05 | 1.01 | True | 0.9239 | 0.560 | 0.8435 | 0.7335 | 0.7847 |
| 7 | 0.10 | 0.95 | True | 0.9227 | 0.530 | 0.8501 | 0.7268 | 0.7836 |
| 8 | 0.25 | 1.01 | False | 0.9214 | 0.520 | 0.8451 | 0.7555 | 0.7978 |
| 9 | 0.15 | 1.01 | False | 0.9210 | 0.560 | 0.8402 | 0.7544 | 0.7950 |
| 10 | 0.30 | 1.01 | False | 0.9202 | 0.510 | 0.8402 | 0.7533 | 0.7944 |

## Conclusion

- The OOF-selected candidate configuration is written to `configs/inference/demo_roi_area_gate_oof_protocol.yml`.
- If the subsequent single external review does not outperform the current mainline, the candidate should be retained as a research record and not merged into the default demo.
- If the subsequent single external review does outperform the current mainline, it is still recommended to confirm through the team's internal validation process before overwriting `demo.yml`.
