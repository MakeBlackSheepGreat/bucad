# Tiny Recommended-Recipe Comparison

Date: 2026-04-25

## Purpose

Retrain `swin_tiny_patch4_window7_224` and `convnext_tiny` with a timm-aware recipe and compare their BUSI external performance against the two current ensemble baselines.

## Recipe Change

| Item | Previous Quick Recipe | New timm-aware Recipe |
| --- | --- | --- |
| Pretrained normalization | `/255` only | timm mean/std normalization |
| Resize/interpolation | direct square resize | timm bicubic + crop_pct |
| Training schedule | fixed learning rate | warmup + cosine decay |
| Checkpoint | last checkpoint | best validation AUC checkpoint |
| Class imbalance | no explicit balancing | balanced class weights |
| Epochs | 20 | 30 |

This run still keeps the project ultrasound preprocessing choice of CLAHE and horizontal flip augmentation. BUSI remains external evaluation only and is not used for training.

## BUSBRA Fold-1 Validation

| Model | Recipe | Best Epoch | AUC | Sensitivity | Specificity | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny | quick project recipe | 16 | 0.7950 | 0.8852 | 0.5020 | 0.6267 |
| Swin-Tiny | timm-aware recipe | 30 | 0.9053 | 0.7541 | 0.8933 | 0.8480 |
| ConvNeXt-Tiny | quick project recipe | 3 | 0.4519 | 0.0000 | 1.0000 | 0.6747 |
| ConvNeXt-Tiny | timm-aware recipe | 13 | 0.9259 | 0.7213 | 0.9328 | 0.8640 |

The new recipe fixes the obvious training-collapse problem. ConvNeXt-Tiny improves from an unusable fold-1 AUC of 0.4519 to 0.9259, and Swin-Tiny improves from 0.7950 to 0.9053.

## BUSI External Evaluation: Default Threshold 0.50

| Model/System | AUC | Threshold | Sensitivity | Specificity | Accuracy | Confusion | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Swin-Tiny fold1 timm-aware | 0.8721 | 0.5000 | 0.7048 | 0.8902 | 0.8300 | TN 389 / FP 48 / FN 62 / TP 148 | single fold, no ensemble |
| ConvNeXt-Tiny fold1 timm-aware | 0.8926 | 0.5000 | 0.7762 | 0.8696 | 0.8393 | TN 380 / FP 57 / FN 47 / TP 163 | single fold, no ensemble |
| EfficientNetV2-S 5-fold ensemble + TTA | 0.8997 | 0.5000 | 0.6619 | 0.9382 | 0.8485 | TN 410 / FP 27 / FN 71 / TP 139 | current ensemble baseline |
| EfficientNetV2-S + DenseNet121 mixed ensemble | 0.9052 | 0.5000 | 0.5810 | 0.9542 | 0.8331 | TN 417 / FP 20 / FN 88 / TP 122 | current mixed ensemble baseline |

## BUSI External Evaluation: Best Youden Operating Point

| Model/System | AUC | Threshold | Sensitivity | Specificity | Accuracy | Confusion | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Swin-Tiny fold1 timm-aware | 0.8721 | 0.1800 | 0.7524 | 0.8696 | 0.8315 | TN 380 / FP 57 / FN 52 / TP 158 | single fold, no ensemble |
| ConvNeXt-Tiny fold1 timm-aware | 0.8926 | 0.5400 | 0.7667 | 0.8879 | 0.8485 | TN 388 / FP 49 / FN 49 / TP 161 | single fold, no ensemble |
| EfficientNetV2-S 5-fold ensemble + TTA | 0.8997 | 0.3300 | 0.8000 | 0.8764 | 0.8516 | TN 383 / FP 54 / FN 42 / TP 168 | current ensemble baseline |
| EfficientNetV2-S + DenseNet121 mixed ensemble | 0.9052 | 0.2700 | 0.8095 | 0.8719 | 0.8516 | TN 381 / FP 56 / FN 40 / TP 170 | current mixed ensemble baseline |

## Interpretation

- The previous Tiny results were not representative because the quick recipe did not match timm pretrained assumptions.
- ConvNeXt-Tiny becomes a credible candidate after the recipe fix. Its BUSI AUC reaches 0.8926 as a single fold, close to the EfficientNetV2-S 5-fold ensemble AUC of 0.8997.
- Swin-Tiny also improves substantially, but its BUSI AUC is lower than ConvNeXt-Tiny under this recipe.
- The current mixed EfficientNetV2-S + DenseNet121 ensemble still has the best BUSI AUC among compared systems at 0.9052.
- Because the Tiny models are only single-fold checkpoints, the fair next step is not to replace the final model immediately. The next step is to test whether ConvNeXt-Tiny improves further with five folds or with a low-weight ensemble contribution.

## Decision

- Keep the current mixed ensemble as the runtime model for now.
- Treat ConvNeXt-Tiny timm-aware as a serious follow-up candidate.
- Do not use the old quick-recipe Tiny results for final model judgment.
- If training time allows, run ConvNeXt-Tiny five folds first; Swin-Tiny is lower priority.

## Artifacts

- Swin train config: `configs/classifier/swin_tiny_patch4_window7_224_timm_recipe.yml`
- ConvNeXt train config: `configs/classifier/convnext_tiny_timm_recipe.yml`
- Swin train report: `artifacts/reports/train_cls_swin_tiny_patch4_window7_224_timm_recipe_fold1.json`
- ConvNeXt train report: `artifacts/reports/train_cls_convnext_tiny_timm_recipe_fold1.json`
- Swin BUSI report: `artifacts/reports/busi_swin_tiny_timm_recipe_fold1.json`
- ConvNeXt BUSI report: `artifacts/reports/busi_convnext_tiny_timm_recipe_fold1.json`
