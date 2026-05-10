<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# ROI Cropping and OOF Fusion Experiment Report

Date: 2026-04-25

## Experiment Boundary

- BUSBRA uses ground truth masks to generate ROIs, and trains the fusor via OOF.
- The `oracle` results for BUSI use BUSI ground truth masks, which can only be regarded as the ROI upper bound and cannot be considered as deployable performance.
- The `segmenter` results for BUSI use the existing `segmenter_fold1.pt` predicted masks, which is a more deployment-representative external validation.
- The subsequent `roi_oof_lcc_optimization.md`, without tuning on BUSI, adds `mask_threshold=0.40` and largest connected component ROI post-processing based on BUSBRA segmentation validation, and has been merged into `demo.yml`.

## OOF Fusor

- Features: full-image 2-model probabilities + ROI 2-model probabilities.
- Feature mode: `logit`.
- OOF CV AUC: `0.9223`.
- OOF recommended threshold: `0.55`.

## External Validation Results

| Solution | ROI Source | ROI-only AUC | Stacking AUC | Stacking@OOF Threshold Sens | Stacking@OOF Threshold Spec | Stacking Best Threshold | Stacking Best Sens | Stacking Best Spec | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| oracle | BUSI ground truth mask | 0.9037 | 0.9202 | 0.8286 | 0.8330 | 0.52 | 0.8905 | 0.8032 | Upper bound reference, not deployable |
| segmenter | segmenter predicted mask | 0.9196 | 0.9196 | 0.8429 | 0.8330 | 0.65 | 0.7810 | 0.9153 | Previous deployable proxy, without LCC post-processing |
| segmenter + LCC | segmenter predicted mask + largest connected component | - | 0.9208 | 0.8524 | 0.8169 | 0.51 | - | - | Current `demo.yml` mainline, mask threshold and LCC from BUSBRA segmentation validation |

## ROI Area Distribution

| Solution | Fallback Count | Mean Area Ratio | Median Area Ratio | P10 | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 0 | 0.4333 | 0.3427 | 0.0511 | 0.9957 |
| segmenter | 3 | 0.6274 | 0.6936 | 0.1575 | 1.0000 |

## Conclusions

- Current full-image mainline AUC: `0.9151`.
- segmenter ROI + OOF fusion AUC: `0.9196`, higher than the full-image mainline, indicating the ROI/lesion region guidance direction is effective.
- After adding the `mask_threshold=0.40` and largest connected component post-processing selected by BUSBRA segmentation validation, the current mainline AUC improved to `0.9208`, and Sensitivity improved to `0.8524`.
- The median area ratio of segmenter ROIs is relatively large, indicating the current approach is not a tight lesion crop but rather lesion region guidance with more context.
- The current ROI OOF configuration has parameters and thresholds fixed based on BUSBRA OOF, and has completed BUSI locked external review; since the AUC and Sensitivity after LCC ROI post-processing are both higher than the previous ROI mainline version, it has been merged as the current demo mainline.

## Competition Metric Completeness Note

The official competition metric set is AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score. If an old archived table shows `-` for Precision or F1-Score, the historical summary did not preserve the confusion matrix or raw probabilities needed to reconstruct that value. For locked BUSI operating-point results, use `artifacts/reports/competition_metrics_all_busi_reports.md` as the complete metric source.
