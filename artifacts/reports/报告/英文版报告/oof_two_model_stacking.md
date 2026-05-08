<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Two-Model OOF Stacking Experiment Report

> Data boundary statement: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or hyperparameter tuning.


Date: 2026-04-25

## Experiment Boundary

- The OOF fusion model is trained only using out-of-fold probabilities from BUSBRA five-fold validation samples.
- BUSI is used only for external evaluation, not involved in fusion model training, cross-validation, or threshold selection.
- This report is a bypass optimization experiment and does not modify `configs/inference/demo.yml`.

## Input Views

- `eff_identity`: EfficientNetV2-S five-fold, identity TTA.
- `conv_crop_sweep`: ConvNeXt-Tiny five-fold, crop-sweep TTA, aligned with the current Demo mainline.
- `conv_hflip`: ConvNeXt-Tiny five-fold, hflip TTA, aligned with the previous round AUC candidate.

## OOF Fusion Results

| Approach | Features | CV AUC | OOF Threshold | BUSI AUC | BUSI@OOF Threshold Sens | Precision | F1-Score | BUSI@OOF Threshold Spec | BUSI@OOF Threshold Acc | BUSI Best Youden Threshold | BUSI Best Youden Sens | BUSI Best Youden Spec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_crop_sweep | eff_identity + conv_crop_sweep | 0.9245 | 0.31 | 0.9130 | 0.9429 | - | - | 0.5629 | 0.6862 | 0.45 | 0.8619 | 0.8192 |
| auc_hflip | eff_identity + conv_hflip | 0.9249 | 0.33 | 0.9134 | 0.9333 | - | - | 0.6041 | 0.7110 | 0.45 | 0.7810 | 0.8947 |
| multi_view | eff_identity + conv_crop_sweep + conv_hflip | 0.9249 | 0.46 | 0.9138 | 0.9429 | - | - | 0.5744 | 0.6940 | 0.66 | 0.7952 | 0.8810 |

## External Confusion Matrices

| Approach | BUSI@OOF Threshold | BUSI Best Youden Point |
| --- | --- | --- |
| current_crop_sweep | TN 246 / FP 191 / FN 12 / TP 198 | TN 358 / FP 79 / FN 29 / TP 181 |
| auc_hflip | TN 264 / FP 173 / FN 14 / TP 196 | TN 391 / FP 46 / FN 46 / TP 164 |
| multi_view | TN 251 / FP 186 / FN 12 / TP 198 | TN 385 / FP 52 / FN 43 / TP 167 |

## Conclusion

- The highest BUSI AUC this round is `multi_view`, with AUC `0.9138`.
- OOF Stacking's training pipeline is more principled than directly searching weights on BUSI, but whether to replace the mainline still depends on whether BUSI external metrics truly improve.
- If OOF threshold transfer to BUSI results in a significant drop in sensitivity, the current Demo mainline should be retained, and this result kept as an experiment record.
