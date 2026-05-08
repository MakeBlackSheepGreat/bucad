<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Small-Scale Weak Fold Retraining External Validation Report

> Data boundary statement: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or hyperparameter tuning.


Date: 2026-04-25

## Experiment Objective

This round of experiments performs small-scale retraining on the weak folds in the current two-model mainline, with the goal of improving overall AUC at lower cost.

The mainline remains unchanged:

- Primary model: `ConvNeXt-Tiny`
- Auxiliary model: `EfficientNetV2-S`
- Current Demo config: `configs/inference/demo.yml`
- Current formal inference config: `configs/inference/ensemble_effnet_convnext_optimized.yml`

This round did not modify `demo.yml`, nor replace mainline weights.

## Data Boundary

- BUSBRA: Used for training and internal five-fold validation.
- BUSI: Used only for external evaluation, not involved in training, not involved in weight selection.
- All retrained checkpoints are saved with new filenames and do not overwrite existing mainline checkpoints.

## Weak Fold Identification

| Model | Fold | Original Internal AUC | Original Sensitivity | Original Specificity | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny | 4 | 0.8875 | 0.7623 | 0.8577 | Lowest among ConvNeXt five folds |
| EfficientNetV2-S | 5 | 0.8523 | 0.6116 | 0.9016 | Lowest among EfficientNet five folds |

## Retraining Configuration

| Config | Target | Output Checkpoint |
| --- | --- | --- |
| `configs/classifier/convnext_tiny_timm_recipe_seed123.yml` | Same training recipe, only changing seed, retrain ConvNeXt-Tiny fold4 | `artifacts/checkpoints/convnext_tiny_timm_recipe_seed123_fold4.pt` |
| `configs/classifier/efficientnetv2_s_bestauc_seed123.yml` | seed123 with best_auc strategy, retrain EfficientNetV2-S fold5 | `artifacts/checkpoints/efficientnetv2_s_bestauc_seed123_fold5.pt` |

## Internal Validation Results

| Model | Fold | Original Internal AUC | Retrained Internal AUC | Change | Observation |
| --- | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny | 4 | 0.8875 | 0.9118 | +0.0243 | Significant improvement in internal validation |
| EfficientNetV2-S | 5 | 0.8523 | 0.9007 | +0.0484 | Significant improvement in internal validation |

Looking only at BUSBRA internal validation, this round of retraining appears effective. However, this is not the final judgment criterion; BUSI external generalization must be examined.

## BUSI Single-Fold External Validation

| Model | Fold | Original BUSI AUC | Retrained BUSI AUC | Change | Conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny crop-sweep | 4 | 0.8901 | 0.8845 | -0.0056 | Internal improvement did not transfer to external |
| EfficientNetV2-S identity | 5 | 0.8633 | 0.8527 | -0.0106 | Internal improvement did not transfer to external |

Single-fold external evaluation already indicates: weak fold retraining better fits the corresponding BUSBRA validation fold, but external domain ranking ability on BUSI decreases.

## BUSI Ensemble External Validation

| Approach | Config | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current Mainline | `configs/inference/ensemble_effnet_convnext_optimized.yml` | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |
| Replace ConvNeXt fold4 only | `configs/inference/ensemble_effnet_convnext_retrain_conv4.yml` | 0.8712 | 0.399 | 0.7571 | 0.8398 | 0.8130 | 0.6943 | 0.7244 | TN 367 / FP 70 / FN 51 / TP 159 |
| Replace EfficientNet fold5 only | `configs/inference/ensemble_effnet_convnext_retrain_eff5.yml` | 0.8708 | 0.399 | 0.7762 | 0.8558 | 0.8300 | 0.7212 | 0.7477 | TN 374 / FP 63 / FN 47 / TP 163 |
| Replace both weak folds | `configs/inference/ensemble_effnet_convnext_retrain_conv4_eff5.yml` | 0.8701 | 0.399 | 0.7524 | 0.8581 | 0.8238 | 0.7182 | 0.7349 | TN 375 / FP 62 / FN 52 / TP 158 |
| hflip candidate with both weak folds replaced | `configs/inference/ensemble_effnet_convnext_hflip_weight0511_retrain_conv4_eff5.yml` | 0.9140 | 0.385 | 0.7571 | 0.8970 | 0.8516 | 0.7794 | 0.7681 | TN 392 / FP 45 / FN 51 / TP 159 |

Ensemble results further confirm: retrained weak folds are not suitable for merging into the current mainline.

## Overfitting Assessment

This round's observations are consistent with typical external generalization overfitting:

1. BUSBRA internal validation AUC significantly increased.
2. BUSI single-fold external AUC decreased.
3. After substitution into the ensemble, overall external AUC significantly decreased.
4. Retrained weights appear to have fitted the local distribution of BUSBRA folds rather than learning more robust cross-dataset features.

Therefore, "fixing the internal weak fold" cannot be used as grounds for replacing the mainline.

## Decision

- Retain the current mainline weights.
- Do not merge `convnext_tiny_timm_recipe_seed123_fold4.pt` into Demo.
- Do not merge `efficientnetv2_s_bestauc_seed123_fold5.pt` into Demo.
- This round's retraining configuration and report are retained as negative result evidence, to be used subsequently to explain why internal validation metrics were not blindly pursued.

## Subsequent Optimization Directions

The next step is recommended to shift toward ROI cropping/lesion region guidance, combined with the OOF mechanism to reduce overfitting risk:

1. **ROI cropping/lesion region guidance**: Use segmentation masks, weak localization, or Grad-CAM to generate tumor region cropped images, reducing the model's dependence on background noise.
2. **Original image + ROI dual-view fusion**: Retain the complete ultrasound image context while adding localized lesion detail input.
3. **OOF mechanism-constrained selection**: All fusion weights, thresholds, and second-level fusion models should be determined on BUSBRA OOF first, with BUSI used only for external validation.
4. **External generalization priority**: Subsequently, do not replace weights based solely on BUSBRA fold metrics; BUSI single-model and ensemble external performance must be checked simultaneously.

## Summary

This round of small-scale retraining yielded an important conclusion: the current project's main bottleneck is not insufficient training of a particular weak fold, but rather the external distribution transfer from BUSBRA to BUSI. Continuing to simply retrain weak folds is not cost-effective; subsequent efforts should focus on input information quality and generalization constraints.
