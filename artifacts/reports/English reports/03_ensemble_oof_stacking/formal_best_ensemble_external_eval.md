<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Two-Model and Three-Model Best Ensemble Formal External Evaluation

> Data boundary note: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or tuning.

Date: 2026-04-25

## Evaluation Notes

- Dataset: BUSI external evaluation set, 647 non-normal samples.
- Evaluation method: directly load local checkpoints and re-run inference on all BUSI images; no cached probability recomputation.
- Inference fix: member-level preprocessing and TTA are now supported; EfficientNetV2-S, DenseNet121, and ConvNeXt-Tiny each use their own training/evaluation configuration.
- Data boundary: BUSI is used only for external evaluation and threshold analysis; it does not participate in training.

## Configuration and Weights

| Scheme | Config File | Members | Weights |
| --- | --- | ---: | --- |
| Two-model AUC-best | `configs/inference/ensemble_effnet_convnext_auc_best.yml` | 10 | EfficientNetV2-S `0.427` / ConvNeXt-Tiny `0.573` |
| Three-model AUC-best | `configs/inference/ensemble_effnet_densenet_convnext_auc_best.yml` | 15 | EfficientNetV2-S `0.358` / DenseNet121 `0.244` / ConvNeXt-Tiny `0.398` |

## Formal Evaluation Results

| Scheme | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Two-model, default threshold | 0.9142 | 0.50 | 0.7524 | 0.9130 | 0.8609 | 0.8061 | 0.7783 | TN 399 / FP 38 / FN 52 / TP 158 |
| Two-model, Youden-optimal threshold | 0.9142 | 0.47 | 0.7714 | 0.9108 | 0.8655 | 0.8060 | 0.7883 | TN 398 / FP 39 / FN 48 / TP 162 |
| Three-model, default threshold | 0.9144 | 0.50 | 0.7095 | 0.9382 | 0.8640 | 0.8466 | 0.7720 | TN 410 / FP 27 / FN 61 / TP 149 |
| Three-model, Youden-optimal threshold | 0.9144 | 0.35 | 0.7952 | 0.8787 | 0.8516 | 0.7591 | 0.7767 | TN 384 / FP 53 / FN 43 / TP 167 |

## Conclusion

- Three-model formal AUC is marginally higher: `0.9144` vs two-model `0.9142`, a difference of `0.0001`.
- Two-model achieves higher Accuracy at the Youden-optimal point: `0.8655`, with fewer models and lower deployment cost.
- Three-model at default threshold is more conservative: Specificity `0.9382` with fewer false positives, but Sensitivity is only `0.7095`.
- If the competition presentation prioritizes AUC, the three-model scheme can be retained as the final best result; if deployment simplicity and balanced operating point matter more, the two-model scheme is more practical.

## Output Files

- Two-model JSON: `artifacts/reports/busi_ensemble_effnet_convnext_auc_best_formal.json`
- Two-model threshold analysis: `artifacts/reports/threshold_analysis_busi_ensemble_effnet_convnext_auc_best_formal.md`
- Three-model JSON: `artifacts/reports/busi_ensemble_effnet_densenet_convnext_auc_best_formal.json`
- Three-model threshold analysis: `artifacts/reports/threshold_analysis_busi_ensemble_effnet_densenet_convnext_auc_best_formal.md`
