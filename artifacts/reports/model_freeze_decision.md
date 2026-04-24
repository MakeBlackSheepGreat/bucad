# Model And Threshold Freeze Decision

Date: 2026-04-24

## Decision

- Frozen runtime classifier: five-fold EfficientNetV2-S ensemble.
- Runtime checkpoints:
  - `artifacts/checkpoints/efficientnetv2_s_fold1.pt`
  - `artifacts/checkpoints/efficientnetv2_s_fold2.pt`
  - `artifacts/checkpoints/efficientnetv2_s_fold3.pt`
  - `artifacts/checkpoints/efficientnetv2_s_fold4.pt`
  - `artifacts/checkpoints/efficientnetv2_s_fold5.pt`
- Frozen runtime segmenter: `artifacts/checkpoints/segmenter_fold1.pt`.
- UI/runtime default threshold: `0.25` in `configs/inference/demo.yml`.
- Evaluation note: `evaluate_busi_dataset` still stores the conventional 0.50 metrics under `metrics`; the selected operating point is recorded under `threshold_analysis.best_by_youden`.

## Evidence

| Evidence | File | Key Result |
| --- | --- | --- |
| EfficientNetV2-S 5-fold summary | `artifacts/reports/efficientnetv2_s_5fold_summary.md` | mean AUC=0.8946, mean sensitivity=0.6525, mean specificity=0.9054 |
| BUSI final external evaluation | `artifacts/reports/busi_eval_final.json` | AUC=0.8955; at threshold 0.25 sensitivity=0.8476, specificity=0.8215, accuracy=0.8300 |
| Release config | `configs/inference/demo.yml` | EfficientNetV2-S ensemble and threshold 0.25 |

## Target Gap

- AUC target >= 0.75: PASS by +0.1455 on BUSI.
- Sensitivity target 0.85: NEAR PASS at threshold 0.25; current sensitivity is 0.8476, short by 0.0024.
- Specificity remains acceptable for demo/report framing at 0.8215 under the selected high-sensitivity threshold.

## Limitations

- T060 full model comparison is still pending, so final report should say EfficientNetV2-S is the selected/frozen model, not yet fully proven best against every baseline.
- BUSI sensitivity is very close to but still slightly below the target; avoid overclaiming clinical screening readiness.
- Grad-CAM and segmentation remain explanatory aids, not clinical ground truth.
## Full Comparison Result

- T060 completed with 7 configured models on fold 1 for 20 epochs.
- Best completed model by AUC: `tf_efficientnetv2_s` with AUC `0.8937`.
- Strong baselines: `densenet121` AUC `0.8867`, `resnet18` AUC `0.8756`, `mobilenetv3_small_100` AUC `0.8704`.
- `alexnet` failed during the recorded run and is documented in `artifacts/reports/comparison_results.json`; torchvision fallback support was added afterward for future reruns.
- Conclusion: the completed comparison supports the EfficientNetV2-S freeze decision.

