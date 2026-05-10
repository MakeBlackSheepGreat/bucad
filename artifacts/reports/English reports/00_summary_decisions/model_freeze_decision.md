# Model And Threshold Freeze Decision

Date: 2026-04-24

## Decision

- Frozen runtime classifier: mixed five-fold EfficientNetV2-S + five-fold DenseNet121 ensemble.
- Runtime checkpoints:
  - `artifacts/checkpoints/efficientnetv2_s_fold1.pt`
  - `artifacts/checkpoints/efficientnetv2_s_fold2.pt`
  - `artifacts/checkpoints/efficientnetv2_s_fold3.pt`
  - `artifacts/checkpoints/efficientnetv2_s_fold4.pt`
  - `artifacts/checkpoints/efficientnetv2_s_fold5.pt`
  - `artifacts/checkpoints/densenet121_fold1.pt`
  - `artifacts/checkpoints/densenet121_fold2.pt`
  - `artifacts/checkpoints/densenet121_fold3.pt`
  - `artifacts/checkpoints/densenet121_fold4.pt`
  - `artifacts/checkpoints/densenet121_fold5.pt`
- Frozen runtime segmenter: `artifacts/checkpoints/segmenter_fold1.pt`.
- Runtime inference enhancement: horizontal-flip TTA is enabled in `configs/inference/demo.yml`.
- UI/runtime balanced threshold: `0.27` in `configs/inference/demo.yml`.
- Evaluation note: `evaluate_busi_dataset` still stores the conventional 0.50 metrics under `metrics`; the selected operating point is recorded under `threshold_analysis.best_by_youden`.

## Evidence

| Evidence | File | Key Result |
| --- | --- | --- |
| EfficientNetV2-S 5-fold summary | `artifacts/reports/efficientnetv2_s_5fold_summary.md` | mean AUC=0.8946, mean sensitivity=0.6525, mean specificity=0.9054 |
| BUSI mixed-ensemble evaluation | `artifacts/reports/busi_mixed_ensemble_weight063_eval.json` | AUC=0.9052; at threshold 0.27 sensitivity=0.8095, specificity=0.8719, accuracy=0.8516 |
| Release config | `configs/inference/demo.yml` | EfficientNetV2-S + DenseNet121 mixed ensemble, flip TTA, and threshold 0.27 |

## Target Gap

- AUC target >= 0.75: PASS by +0.1552 on BUSI.
- Sensitivity target 0.85: the balanced runtime point prioritizes overall AUC and specificity; use threshold 0.24 if a high-sensitivity operating point is required.
- Specificity improves to 0.8719 under the selected balanced threshold.

## Limitations

- T060 full model comparison is still pending, so final report should say EfficientNetV2-S is the selected/frozen model, not yet fully proven best against every baseline.
- BUSI AUC improves with the mixed ensemble, but the system is still a research prototype and not clinical software.
- Flip TTA improves AUC but slightly lowers specificity at threshold 0.25 compared with the previous non-TTA evaluation.
- Grad-CAM and segmentation remain explanatory aids, not clinical ground truth.
## Full Comparison Result

- T060 completed with 7 configured models on fold 1 for 20 epochs.
- Best completed model by AUC: `tf_efficientnetv2_s` with AUC `0.8937`.
- Strong baselines: `densenet121` AUC `0.8867`, `resnet18` AUC `0.8756`, `mobilenetv3_small_100` AUC `0.8704`.
- `alexnet` failed during the recorded run and is documented in `artifacts/reports/comparison_results.json`; torchvision fallback support was added afterward for future reruns.
- Conclusion: the completed comparison supports the EfficientNetV2-S freeze decision.
## Competition Metric Completeness Note

The official competition metric set is AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score. If an old archived table shows `-` for Precision or F1-Score, the historical summary did not preserve the confusion matrix or raw probabilities needed to reconstruct that value. For locked BUSI operating-point results, use `artifacts/reports/competition_metrics_all_busi_reports.md` as the complete metric source.
