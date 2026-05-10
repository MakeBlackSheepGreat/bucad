<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# ConvNeXt-Small Strong Regularization and 5-Fold Validation Experiment Report

> Data boundary note: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or tuning.


Date: 2026-04-25

## Experiment Boundary

- This report only records the bypass test for the main model upgrade. It does not modify `configs/inference/demo.yml` or change the current Demo mainline.
- Training still strictly uses only the BUSBRA training/internal validation split; BUSI is used only for external evaluation, threshold analysis, and error observation.
- Comparisons distinguish between "single-model / 5-fold / ensemble" categories to avoid conflating unoptimized single models with already-optimized formal ensemble solutions.

## Objectives of This Round

1. Verify whether ConvNeXt-Small can serve as a stronger main model than ConvNeXt-Tiny.
2. Verify whether stronger regularization can mitigate overfitting caused by the increased parameter count of Small.
3. After extending to 5 folds, observe whether Small's overall generalization capability can surpass the current mainline.
4. Offline check whether Small can serve as a low-weight auxiliary model providing complementary benefits to the existing ensemble.

## Configuration Overview

| Config | File | Description |
| --- | --- | --- |
| Small base training | `configs/classifier/convnext_small_timm_recipe.yml` | ConvNeXt-Small, following the timm recommended preprocessing approach, serving as the upgrade main model baseline |
| Small strong regularization training | `configs/classifier/convnext_small_timm_recipe_regularized.yml` | Reduced learning rate, increased weight decay, drop path, label smoothing, and light augmentation |
| Small fold1 TTA | `configs/inference/convnext_small_timm_recipe_fold1_tta_crop_sweep.yml` | fold1 single model crop-sweep TTA |
| Small 5-fold TTA | `configs/inference/convnext_small_timm_recipe_5fold_tta_crop_sweep.yml` | 5-fold averaging + crop-sweep TTA |

## Internal 5-Fold Validation Results

| Model | Fold | AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Small | 1 | 0.9118 | 0.7295 | - | - | 0.9091 | 0.8507 | 28 |
| ConvNeXt-Small | 2 | 0.9177 | 0.8926 | - | - | 0.8150 | 0.8400 | 8 |
| ConvNeXt-Small | 3 | 0.9444 | 0.7603 | - | - | 0.8976 | 0.8533 | 10 |
| ConvNeXt-Small | 4 | 0.8936 | 0.8279 | - | - | 0.8458 | 0.8400 | 4 |
| ConvNeXt-Small | 5 | 0.9135 | 0.7934 | - | - | 0.8976 | 0.8640 | 16 |
| ConvNeXt-Small Mean | 1-5 | 0.9162 | 0.8007 | - | - | 0.8730 | 0.8496 | - |

The mean internal 5-fold performance is not poor, but fold4 is notably lower, and the best epochs for multiple folds are relatively early, indicating that the larger model does indeed tend to enter the overfitting zone earlier at the current data scale.

## BUSI External Evaluation Results

| Solution | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Small fold1 crop-sweep | 0.8991 | 0.50 | 0.7857 | 0.8673 | 0.8408 | 0.7399 | 0.7621 | TN 379 / FP 58 / FN 45 / TP 165 |
| ConvNeXt-Small fold1 strong regularization crop-sweep | 0.8759 | 0.50 | 0.7714 | 0.8650 | 0.8346 | 0.7330 | 0.7517 | TN 378 / FP 59 / FN 48 / TP 162 |
| ConvNeXt-Small 5-fold crop-sweep | 0.9040 | 0.50 | 0.8190 | 0.8535 | 0.8423 | 0.7288 | 0.7713 | TN 373 / FP 64 / FN 38 / TP 172 |
| ConvNeXt-Tiny 5-fold crop-sweep | 0.9054 | 0.50 | 0.7714 | 0.8879 | 0.8501 | 0.7678 | 0.7696 | TN 388 / FP 49 / FN 48 / TP 162 |
| Current 2-model formal solution | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |
| Current 3-model optimized solution | 0.9162 | 0.453 | 0.7476 | 0.9291 | 0.8702 | 0.8351 | 0.7889 | TN 406 / FP 31 / FN 53 / TP 157 |

## Threshold Observation

The Youden-optimal threshold for Small 5-fold on BUSI is `0.47`:

| Threshold Strategy | Threshold | Sensitivity | Specificity | Accuracy | Youden J |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default threshold | 0.50 | 0.8190 | 0.8535 | 0.8423 | 0.6726 |
| Youden-optimal | 0.47 | 0.8333 | 0.8467 | 0.8423 | 0.6800 |
| Accuracy-leaning point | 0.58 | 0.7810 | 0.8879 | 0.8532 | 0.6688 |

The main characteristic of Small 5-fold is better sensitivity, but lower specificity than the current mainline. It can reduce missed diagnoses but at the cost of more false positives.

## Offline Probability Fusion Observation

Offline weighted search based on existing BUSI output probabilities, used only for assessing complementarity, not as formal training or final conclusions.

| Fusion Method | Small Weight | AUC | Recommended Threshold | Sensitivity | Precision | F1-Score | Specificity | Accuracy | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2-model formal solution + Small | 0.17 | 0.9153 | 0.401 | 0.8095 | - | - | 0.8719 | 0.8516 | AUC improved by only about 0.0002, very weak benefit |
| 3-model optimized solution + Small | 0.20 | 0.9165 | 0.343 | 0.8333 | - | - | 0.8535 | 0.8470 | AUC improved by only about 0.0003, but significantly sacrifices specificity |
| Tiny 5-fold + Small | 0.43 | 0.9085 | 0.391 | 0.8524 | - | - | 0.8330 | 0.8393 | More recall-oriented, overall inferior to the formal ensemble |

Small shows some complementarity, but the complementary value mainly manifests in improving Sensitivity rather than stably improving AUC. The current evidence is insufficient to support promoting Small to a formal main model.

## Conclusions

- The strong regularization version did not bring benefits: fold1 external AUC dropped from the base Small's `0.8991` to `0.8759`, indicating that the current regularization strength or training strategy is not suitable for direct scaling.
- Small 5-fold is more stable than Small fold1, with external AUC improving to `0.9040`, but still lower than ConvNeXt-Tiny 5-fold `0.9054` and the current formal 2-model solution `0.9151`.
- Small's advantage is Sensitivity, reaching `0.8190` at default threshold, but at the cost of Specificity dropping to `0.8535`.
- The current mainline is not recommended for replacement. ConvNeXt-Tiny + EfficientNetV2-S remains the more stable Demo main solution.

## Recommended Next Steps

1. Do not continue blindly increasing ConvNeXt-Small regularization; it already shows signals of "good internal performance, declining external performance."
2. If the goal is to improve AUC, prioritize seed ensemble, OOF calibration, or lightweight stacking for the current 2-model mainline, rather than continuing to scale up the single model.
3. If the goal is to improve Sensitivity, Small can be retained as a low-weight auxiliary branch or provided as a separate "high-recall mode," but it must be made clear that this sacrifices specificity.
4. More worthwhile experiments to pursue next include BUSBRA OOF-based threshold/temperature calibration, hard case review, ROI/segmentation-assisted cropping, rather than continuing to simply stack model capacity.
