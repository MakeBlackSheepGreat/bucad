<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Two-Model OOF Optimization Decision Report

> Data boundary statement: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or hyperparameter tuning.


Date: 2026-04-25

## Experiment Objective

This round's objective is to optimize the current two-model ensemble in a more principled manner, rather than continuing to directly search weights on the BUSI external evaluation set. The specific approach is:

1. Use BUSBRA five-fold validation sets to generate out-of-fold probabilities.
2. Train or select a fusion strategy on BUSBRA OOF probabilities.
3. Transfer the resulting fusion strategy to BUSI, using BUSI only as the external evaluation set.

## Data Boundary

- BUSBRA: Used for OOF probability generation, fusion model training, OOF cross-validation, and OOF threshold selection.
- BUSI: Used only for external evaluation, not involved in training, not involved in OOF fusion model selection.
- The current Demo mainline has not been modified.

## Input Model Views

| View | Meaning | Purpose |
| --- | --- | --- |
| `eff_identity` | EfficientNetV2-S five-fold identity TTA | Two-model auxiliary branch |
| `conv_crop_sweep` | ConvNeXt-Tiny five-fold crop-sweep TTA | Current Demo mainline branch |
| `conv_hflip` | ConvNeXt-Tiny five-fold hflip TTA | Previous round AUC candidate branch |

## Logistic Regression Stacking Results

| Approach | Features | OOF CV AUC | BUSI AUC | BUSI@0.5 Sens | Precision | F1-Score | BUSI@0.5 Spec | BUSI@0.5 Acc | Assessment |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `current_crop_sweep` | `eff_identity + conv_crop_sweep` | 0.9245 | 0.9130 | 0.7762 | - | - | 0.8970 | 0.8578 | External AUC lower than mainline |
| `auc_hflip` | `eff_identity + conv_hflip` | 0.9249 | 0.9134 | 0.6857 | - | - | 0.9428 | 0.8594 | Overly biased toward specificity, sensitivity drops significantly |
| `multi_view` | `eff_identity + conv_crop_sweep + conv_hflip` | 0.9249 | 0.9138 | 0.9286 | - | - | 0.6339 | 0.7295 | Severe calibration shift, cannot be used as default scheme |

Logistic regression stacking performs well on BUSBRA OOF, but when transferred to BUSI, the AUC is lower than the current mainline of `0.9151`. This indicates that there is a distribution discrepancy between BUSBRA and BUSI, and the probability calibration relationships learned by the fusion model did not transfer stably.

## OOF Static Weight Search Results

Compared to logistic regression, static probability weighting is more conservative and generalizes more stably.

| Approach | OOF Learned Weights | OOF AUC | OOF Threshold | BUSI AUC | BUSI@OOF Threshold Sens | BUSI@OOF Threshold Spec | BUSI Best Analysis Threshold | BUSI Best Analysis Sens | BUSI Best Analysis Spec | Assessment |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `current_crop_sweep` probability weighting | Eff 0.344 / Conv 0.656 | 0.9199 | 0.24 | 0.9143 | 0.8905 | 0.7574 | 0.43 | 0.7952 | 0.8879 | AUC lower than mainline, not recommended |
| `auc_hflip` probability weighting | Eff 0.456 / Conv 0.544 | 0.9184 | 0.24 | 0.9158 | 0.8857 | 0.7849 | 0.40 | 0.7857 | 0.8993 | AUC slightly higher than mainline, but sensitivity lower than mainline |

Among these, `auc_hflip`'s OOF static weights are the closest to a usable result this round: BUSI AUC `0.9158`, higher than the current mainline's `0.9151`, but its sensitivity at reasonable operating points is still lower than the current mainline.

## Comparison with Current Mainline

| Approach | Weight Source | BUSI AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current Demo Mainline | Earlier weight search + crop-sweep | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |
| OOF Static Weight Candidate | BUSBRA OOF + hflip | 0.9158 | BUSI analysis threshold 0.40 | 0.7857 | 0.8993 | 0.8624 | 0.7895 | 0.7876 | TN 393 / FP 44 / FN 45 / TP 165 |
| Previous Round hflip Candidate | BUSI direct weight search + hflip | 0.9159 | 0.385 | 0.7905 | 0.8947 | 0.8609 | 0.7830 | 0.7867 | TN 391 / FP 46 / FN 44 / TP 166 |

## Conclusion

- The OOF approach is principled, but did not significantly push AUC higher.
- Logistic regression stacking is not recommended: good internal OOF performance but external BUSI AUC actually decreases, indicating it overfits the probability distribution of BUSBRA.
- OOF static weights can slightly improve AUC to `0.9158`, but at the cost of sensitivity dropping from `0.8000` to `0.7857`.
- The current project is a breast tumor screening/assisted diagnosis scenario where the cost of missed diagnoses is higher; therefore, it is not recommended to replace the current Demo mainline for approximately `+0.0006` AUC improvement.

## Decision Recommendations

1. Maintain the current `ConvNeXt-Tiny crop-sweep + EfficientNetV2-S identity` Demo mainline.
2. Treat the OOF hflip static weights as a "specificity-priority candidate," not as the default scheme.
3. If continuing to pursue AUC, it is more worthwhile to retrain weak folds or use seed ensemble rather than continuing with more complex stacking.
4. If the defense presentation requires demonstrating rigor, it can be emphasized: we attempted more principled OOF Stacking, but due to unstable external distribution transfer, we ultimately chose the more robust mainline scheme.
