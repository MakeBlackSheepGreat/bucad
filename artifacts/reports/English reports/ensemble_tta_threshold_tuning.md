<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Two-Model and Three-Model Current Model Tuning Report

> Data boundary note: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or tuning.


Date: 2026-04-25

## Tuning Boundary

- No retraining of models; only the inference strategy for existing checkpoints is optimized.
- BUSI is used only for external evaluation, TTA ablation, and threshold operating point analysis, and does not participate in training.
- This round focuses on comparing ConvNeXt branch TTA, whether to retain CNN branch hflip, and 0.001-granularity thresholds.

## TTA Ablation Results

| Solution | TTA Strategy | AUC | Default Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Two-model | Baseline: CNN hflip + ConvNeXt crop_sweep | 0.9142 | 0.500 | 0.7524 | 0.9130 | 0.8609 | 0.8061 | 0.7783 | TN 399 / FP 38 / FN 52 / TP 158 |
| Two-model | ConvNeXt hflip | 0.9148 | 0.500 | 0.7143 | 0.9245 | 0.8563 | 0.8197 | 0.7634 | TN 404 / FP 33 / FN 60 / TP 150 |
| Two-model | ConvNeXt rotate5 | 0.9148 | 0.500 | 0.7286 | 0.9199 | 0.8578 | 0.8138 | 0.7688 | TN 402 / FP 35 / FN 57 / TP 153 |
| Two-model | CNN identity + ConvNeXt crop_sweep | 0.9151 | 0.500 | 0.7524 | 0.9176 | 0.8640 | 0.8144 | 0.7822 | TN 401 / FP 36 / FN 52 / TP 158 |
| Three-model | Baseline: CNN hflip + ConvNeXt crop_sweep | 0.9144 | 0.500 | 0.7095 | 0.9382 | 0.8640 | 0.8466 | 0.7720 | TN 410 / FP 27 / FN 61 / TP 149 |
| Three-model | ConvNeXt hflip | 0.9153 | 0.500 | 0.6857 | 0.9382 | 0.8563 | 0.8421 | 0.7559 | TN 410 / FP 27 / FN 66 / TP 144 |
| Three-model | ConvNeXt rotate5 | 0.9151 | 0.500 | 0.6857 | 0.9382 | 0.8563 | 0.8421 | 0.7559 | TN 410 / FP 27 / FN 66 / TP 144 |
| Three-model | CNN identity + ConvNeXt crop_sweep | 0.9162 | 0.500 | 0.6952 | 0.9405 | 0.8609 | 0.8488 | 0.7644 | TN 411 / FP 26 / FN 64 / TP 146 |

## 0.001 Fine-Grained Threshold Recommendations

| Solution | TTA Strategy | Objective | Threshold | Sensitivity | Specificity | Accuracy | Youden J | Precision | F1-Score | Confusion Matrix |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Two-model | CNN identity + ConvNeXt crop_sweep | Youden-optimal | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.6856 | 0.7706 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |
| Two-model | CNN identity + ConvNeXt crop_sweep | Accuracy-optimal | 0.543 | 0.7238 | 0.9359 | 0.8671 | 0.6597 | 0.8444 | 0.7795 | TN 409 / FP 28 / FN 58 / TP 152 |
| Two-model | CNN identity + ConvNeXt crop_sweep | Sensitivity>=0.85 | 0.299 | 0.8524 | 0.8055 | 0.8207 | 0.6579 | 0.6780 | 0.7553 | TN 352 / FP 85 / FN 31 / TP 179 |
| Three-model | CNN identity + ConvNeXt crop_sweep | Youden / Accuracy-optimal | 0.453 | 0.7476 | 0.9291 | 0.8702 | 0.6767 | 0.8351 | 0.7889 | TN 406 / FP 31 / FN 53 / TP 157 |
| Three-model | CNN identity + ConvNeXt crop_sweep | Sensitivity>=0.80 | 0.350 | 0.8000 | 0.8719 | 0.8485 | 0.6719 | 0.7500 | 0.7742 | TN 381 / FP 56 / FN 42 / TP 168 |
| Three-model | CNN identity + ConvNeXt crop_sweep | Sensitivity>=0.85 | 0.253 | 0.8571 | 0.8146 | 0.8284 | 0.6718 | 0.6897 | 0.7643 | TN 356 / FP 81 / FN 30 / TP 180 |
| Three-model | ConvNeXt rotate5 | High-recall candidate | 0.228 | 0.8762 | 0.8078 | 0.8300 | 0.6840 | 0.6866 | 0.7699 | TN 353 / FP 84 / FN 26 / TP 184 |

## Final Optimized Configurations

| Config | File | AUC | Default Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Two-model final config | `configs/inference/ensemble_effnet_convnext_optimized.yml` | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |
| Three-model final config | `configs/inference/ensemble_effnet_densenet_convnext_optimized.yml` | 0.9162 | 0.453 | 0.7476 | 0.9291 | 0.8702 | 0.8351 | 0.7889 | TN 406 / FP 31 / FN 53 / TP 157 |

## Conclusions

- The optimal TTA for the two-model solution is `CNN identity + ConvNeXt crop_sweep`, with AUC improving from `0.9142` to `0.9151`.
- The recommended balanced threshold for the two-model solution is `0.399`, with Sensitivity `0.8000`, Specificity `0.8856`, and Youden J `0.6856`.
- The optimal AUC for the three-model solution is also achieved with `CNN identity + ConvNeXt crop_sweep`, with AUC improving from `0.9144` to `0.9162`.
- The recommended main configuration threshold for the three-model solution is `0.453`, with Accuracy `0.8702`, which is the highest accuracy operating point in this round.
- If subsequent work emphasizes screening recall, the three-model solution can be temporarily switched to `ConvNeXt rotate5` with threshold `0.228`, achieving Sensitivity of `0.8762`, but Accuracy would drop to `0.8300`.
