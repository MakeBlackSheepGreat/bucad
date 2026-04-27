# ROI Segmenter 5-Fold Comparison

Date: 2026-04-26 23:29:10

## Decision

- Keep `configs/inference/demo.yml` on the current `segmenter_fold1.pt` runtime segmenter.
- Do not merge `segmenter_5fold_fold5.pt` or the 5-fold mask ensemble into the default demo.
- The 5-fold segmentation run improves validation Dice, but the resulting ROI masks did not improve BUSI diagnostic metrics under the current ROI Area Gate stacker.

## Fixed-Threshold BUSI Results

| Candidate | AUC | Threshold | Accuracy | Sensitivity | Precision | Specificity | F1-Score | Confusion | Delta AUC | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| current_fold1_runtime_recheck | 0.9256 | 0.51 | 0.8532 | 0.8667 | 0.7309 | 0.8467 | 0.7930 | TN 370 / FP 67 / FN 28 / TP 182 | +0.0000 | +0.0000 |
| seg5fold_best_fold5 | 0.9181 | 0.51 | 0.8439 | 0.8524 | 0.7189 | 0.8398 | 0.7800 | TN 367 / FP 70 / FN 31 / TP 179 | -0.0075 | -0.0131 |
| seg5fold_mask_ensemble | 0.9200 | 0.51 | 0.8346 | 0.8524 | 0.7020 | 0.8261 | 0.7699 | TN 361 / FP 76 / FN 31 / TP 179 | -0.0055 | -0.0231 |

## Candidate Setup

- `current_fold1_runtime_recheck`: current `configs/inference/demo.yml`, re-run after adding segmenter-list runtime support.
- `seg5fold_best_fold5`: `configs/inference/demo_roi_area_gate_seg5fold_best.yml`, selected by BUSBRA segmentation Dice before BUSI evaluation.
- `seg5fold_mask_ensemble`: `configs/inference/demo_roi_area_gate_seg5fold_ensemble.yml`, averages sigmoid masks from `segmenter_5fold_fold1.pt` through `segmenter_5fold_fold5.pt`.

## Interpretation

- Fold5 improved segmentation validation Dice, but its BUSI ROI diagnostic AUC dropped by `0.0075` and F1 dropped by `0.0131` compared with the current mainline.
- The 5-fold mask ensemble also underperformed the current mainline, with AUC down `0.0055` and F1 down `0.0231`.
- This suggests the current ROI stacker and area gate are tuned to the existing segmenter mask distribution; better Dice alone is not sufficient to improve diagnostic ROI fusion.
- A future improvement should retrain or recalibrate the ROI stacker and area gate with the new segmenter masks on BUSBRA OOF before another locked BUSI check.
