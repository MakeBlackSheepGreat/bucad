<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->
# Competition Metrics Report Regression Check

## Check Background

After user review, it was noted that the README top evaluation results and some historical experiment reports still use the old metrics table, showing only AUC, Sensitivity, Specificity, and Accuracy, missing `Precision` and `F1-Score` required by the competition. The goal of this regression check is to unify the reporting standard, not to retrain models, reselect models, or re-tune parameters.

## Competition Metrics Standard

Subsequent BUSI external evaluation operating points must uniformly display the following metrics:

| Metric | Meaning | Current Report Field |
| --- | --- | --- |
| AUC | Area under curve, measuring the model's overall ability to distinguish benign from malignant samples | `auc` |
| Accuracy | Proportion of correctly predicted samples out of total samples | `accuracy` |
| Recall/Sensitivity | Proportion of actual malignant samples correctly identified as malignant | `sensitivity` / `recall` |
| Precision | Proportion of true malignancies among samples predicted as malignant | `precision` |
| Specificity | Proportion of actual benign samples correctly identified as benign | `specificity` |
| F1-Score | Harmonic mean of Precision and Recall/Sensitivity | `f1_score` |

## Scope of This Correction

- The README top demo mainline results table now includes `Precision` and `F1-Score`, reordered to the full competition metrics sequence.
- The README top three-model non-ROI comparison table now includes `Precision` and `F1-Score`.
- Top-level experiment report tables containing `TN / FP / FN / TP` confusion matrices have been automatically populated with `Precision` and `F1-Score` based on the confusion matrices.
- Archived English/Chinese threshold analysis reports have been regenerated from the corresponding BUSI JSON; default threshold, Youden J optimal threshold, and threshold sweep tables all include complete competition metrics.
- For historical summary tables without saved confusion matrices or raw probabilities, `-` is retained in the report, with a "Competition Metrics Completeness Note" appended to avoid fabricating metrics that cannot be reverse-calculated.

## Data Boundary Statement

This correction only modifies report display fields and does not change any model weights, thresholds, ROI parameters, or ensemble weights. BUSBRA is still used for training, internal validation, OOF selection, ROI parameter selection, and threshold selection; BUSI is still only used for post-configuration-lock external evaluation and does not participate in training or tuning.

## Current Mainline Complete Metrics

| Configuration | AUC | Threshold | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF Stacking + LCC ROI | 0.9208 | 0.550 | 0.8284 | 0.8524 | 0.6911 | 0.8169 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |

## Complete Metrics Sources

- `artifacts/reports/competition_metrics_all_busi_reports.md`: Complete metrics summary of all BUSI JSON reports.
- `artifacts/reports/competition_metrics_audit.md`: Metrics formula and evaluation field audit.
- `artifacts/reports/threshold_analysis.md`: Current mainline threshold sweep.
