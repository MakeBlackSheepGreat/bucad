<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->
# Competition Metrics Review and Re-evaluation Report

Date: 2026-04-26

## Review Conclusions

This review found: the current computation formulas for AUC, Accuracy, Recall/Sensitivity, and Specificity show no bias, but the previous report output was incomplete, missing `Precision` and `F1-Score` required by the competition. These have therefore been supplemented starting from the unified metrics function in `src/utils/metrics.py`, and the BUSI external evaluation JSON and threshold markdown have been regenerated.

It should be emphasized: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI remains the locked external evaluation set. The BUSI threshold sweep in this report is only for operating point analysis and should not be used as the basis for training or tuning. Formal reports should preferentially cite metrics under the demo fixed threshold `0.55`.

## Competition Metrics Mapping

| Competition Metric | Project Field | Computation Meaning | Current Status |
| --- | --- | --- | --- |
| AUC | `auc` | ROC-AUC computed based on malignant probability | Reviewed and output |
| Accuracy | `accuracy` | `(TP + TN) / total samples` | Reviewed and output |
| Recall/Sensitivity | `sensitivity` / `recall` | `TP / (TP + FN)`, malignant detection rate | Reviewed and output |
| Precision | `precision` | `TP / (TP + FP)`, proportion of true malignancies among predicted malignancies | Supplemented and output |
| Specificity | `specificity` | `TN / (TN + FP)`, benign exclusion capability | Reviewed and output |
| F1-Score | `f1_score` | `2 * Precision * Recall / (Precision + Recall)` | Supplemented and output |

## Re-evaluation Configuration

| Item | Content |
| --- | --- |
| Evaluation dataset | BUSI external evaluation only |
| Config | `configs/inference/demo.yml` |
| Output JSON | `artifacts/reports/competition_metrics_demo_eval.json` |
| Threshold report | `artifacts/reports/threshold_analysis_competition_metrics_demo_eval.md` |
| Sample count | `647` |
| Classifier member count | `10` |
| Main model family | ConvNeXt-Tiny + EfficientNetV2-S ensemble with ROI OOF LCC mainline |

## Fixed Demo Threshold Results: threshold 0.55

This table represents the recommended reporting baseline for the current demo mainline configuration, because the threshold comes from the config file and was not re-tuned on BUSI this time.

| AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Threshold |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9208 | 0.8284 | 0.8524 | 0.6911 | 0.8169 | 0.7633 | 0.5500 |

| TN | FP | FN | TP |
| ---: | ---: | ---: | ---: |
| 357 | 80 | 31 | 179 |

## BUSI Youden Analysis Point: threshold 0.51

This table is only for operating point analysis after external evaluation and should not be used as the basis for model training, weight selection, or formal threshold selection.

| AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Threshold | Youden J |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9208 | 0.8269 | 0.9000 | 0.6750 | 0.7918 | 0.7714 | 0.5100 | 0.6918 |

| TN | FP | FN | TP |
| ---: | ---: | ---: | ---: |
| 346 | 91 | 21 | 189 |

## Bias Fix Log

- `src/utils/metrics.py`: Added `precision`, `recall` alias and `f1_score`, while retaining the original `sensitivity` field.
- `src/engine/inference.py`: Default threshold, best Youden, and sweep tables in threshold markdown all now include Precision and F1-Score.
- `src/utils/document_reports.py`: Metrics tables in DOCX/structured reports now include Precision and F1-Score.
- `scripts/search_busi_ensemble_weights.py`: Weight search reuses unified threshold metrics to avoid manual metric omissions.
- `tests/unit/test_metrics.py`, `tests/integration/test_busi_eval.py`: Added Precision/F1 assertions.

## Verification Commands

```powershell
C:\Users\876762330\.conda\envs\BUCAD\python.exe -m pytest tests\unit\test_metrics.py tests\integration\test_busi_eval.py
C:\Users\876762330\.conda\envs\BUCAD\python.exe scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\competition_metrics_demo_eval.json
```

## Verification Results

- Metrics tests: `6 passed in 9.80s`
- Full tests: `59 passed in 11.57s`
- BUSI external evaluation: Completed, output includes AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, F1-Score, and confusion matrix.
- Conclusion: The previous report fields were incomplete and required correction; after correction, competition technical metric requirements are now met.
