<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# ConvNeXt-Small Main Model Upgrade Bypass Experiment

> Data boundary note: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or tuning.


Date: 2026-04-25

## Experiment Boundary

- This experiment only tests the feasibility of upgrading the main model. It does not modify `configs/inference/demo.yml`, README, or the current mainline ensemble solution.
- Training data still only uses the BUSBRA fold1 training split; BUSI is used only for external evaluation and threshold analysis.
- Comparisons use fold1 single models to avoid directly comparing a single-fold model against already-optimized 5-fold/ensemble models.

## New Configurations

- Training config: `configs/classifier/convnext_small_timm_recipe.yml`
- Standard TTA evaluation config: `configs/inference/convnext_small_timm_recipe_fold1.yml`
- Crop-sweep TTA evaluation config: `configs/inference/convnext_small_timm_recipe_fold1_tta_crop_sweep.yml`
- Fair comparison config: `configs/inference/convnext_tiny_timm_recipe_fold1_tta_crop_sweep.yml`

## Training Results

| Model | Fold | Internal Validation AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny timm recipe | 1 | 0.9259 | 0.7213 | - | - | 0.9328 | 0.8640 | Existing baseline |
| ConvNeXt-Small timm recipe | 1 | 0.9118 | 0.7295 | - | - | 0.9091 | 0.8507 | New in this round |

Training observation: ConvNeXt-Small's training loss approached 0 in the later stages, indicating that the increased parameter count makes it easier to memorize the training set, posing an overfitting risk.

## BUSI External Evaluation

| Model | TTA | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny fold1 | hflip | 0.8943 | 0.50 | 0.7762 | 0.8741 | 0.8423 | 0.7477 | 0.7617 | TN 382 / FP 55 / FN 47 / TP 163 |
| ConvNeXt-Tiny fold1 | crop-sweep | 0.8953 | 0.50 | 0.7952 | 0.8696 | 0.8454 | 0.7455 | 0.7696 | TN 380 / FP 57 / FN 43 / TP 167 |
| ConvNeXt-Small fold1 | hflip | 0.8947 | 0.50 | 0.7667 | 0.8581 | 0.8284 | 0.7220 | 0.7436 | TN 375 / FP 62 / FN 49 / TP 161 |
| ConvNeXt-Small fold1 | crop-sweep | 0.8991 | 0.50 | 0.7857 | 0.8673 | 0.8408 | 0.7399 | 0.7621 | TN 379 / FP 58 / FN 45 / TP 165 |

## Youden Operating Points

| Model | TTA | AUC | Recommended Threshold | Sensitivity | Precision | F1-Score | Specificity | Accuracy | Youden J |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny fold1 | crop-sweep | 0.8953 | 0.49 | 0.7952 | - | - | 0.8719 | 0.8470 | 0.6671 |
| ConvNeXt-Small fold1 | crop-sweep | 0.8991 | 0.28 | 0.8381 | - | - | 0.8284 | 0.8315 | 0.6665 |

## Conclusions

- ConvNeXt-Small is not a failed direction: under the fair comparison of fold1 + crop-sweep TTA, BUSI AUC improved from `0.8953` to `0.8991`, an increase of about `0.0037`.
- But it cannot yet replace the current mainline: the current formal demo is a ConvNeXt-Tiny + EfficientNetV2-S 2-model 5-fold optimized ensemble with AUC `0.9151`, which is clearly higher than single-fold Small.
- Small's internal validation AUC is lower than Tiny's, and the training loss approaches 0 in later stages, indicating that larger models need stronger regularization or more conservative training strategies.
- Recommendation: do not change the mainline for now. If further pursued, the next step should be training ConvNeXt-Small across 5 folds, with stronger regularization, lower learning rate, early stopping, or Mixup/CutMix, and then fairly comparing against the current 5-fold Tiny main model.
