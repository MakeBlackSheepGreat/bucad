# ConvNeXt-Tiny Main-Model Optimization

Date: 2026-04-25

## Goal

Optimize ConvNeXt-Tiny as a possible new main classifier, then compare it against the two existing ensemble baselines. BUSBRA remains the only training/internal-validation data source. BUSI is external evaluation only.

## Training Recipe

| Item | Setting |
| --- | --- |
| Backbone | `convnext_tiny` |
| Pretraining | timm pretrained weights |
| Input | 224, timm bicubic interpolation, crop_pct 0.95 |
| Normalization | timm mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| Ultrasound preprocessing | CLAHE enabled |
| Augmentation | horizontal flip |
| Optimizer | AdamW |
| Schedule | warmup + cosine decay |
| Class balance | balanced class weights |
| Checkpoint selection | best validation AUC |
| Folds | 5 |

## BUSBRA Five-Fold Internal Validation

| Fold | Best Epoch | AUC | Sensitivity | Specificity | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13 | 0.9259 | 0.7213 | 0.9328 | 0.8640 |
| 2 | 3 | 0.9301 | 0.9008 | 0.7913 | 0.8267 |
| 3 | 30 | 0.9469 | 0.8099 | 0.9173 | 0.8827 |
| 4 | 14 | 0.8875 | 0.7623 | 0.8577 | 0.8267 |
| 5 | 10 | 0.9158 | 0.6777 | 0.9331 | 0.8507 |
| Mean | - | 0.9212 | 0.7744 | 0.8864 | 0.8501 |

The five-fold internal result is stable enough to treat ConvNeXt-Tiny as a serious main-model candidate. Fold 4 is weaker than the others, but the mean AUC remains above 0.91.

## BUSI External Evaluation: Default Threshold 0.50

| Model/System | AUC | Threshold | Sensitivity | Specificity | Accuracy | Confusion | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ConvNeXt-Tiny fold1 | 0.8926 | 0.5000 | 0.7762 | 0.8696 | 0.8393 | TN 380 / FP 57 / FN 47 / TP 163 | single fold |
| ConvNeXt-Tiny 5-fold ensemble | 0.9044 | 0.5000 | 0.7524 | 0.8947 | 0.8485 | TN 391 / FP 46 / FN 52 / TP 158 | new main-model candidate |
| EfficientNetV2-S 5-fold + TTA | 0.8997 | 0.5000 | 0.6619 | 0.9382 | 0.8485 | TN 410 / FP 27 / FN 71 / TP 139 | existing ensemble baseline |
| EfficientNetV2-S + DenseNet121 mixed ensemble | 0.9052 | 0.5000 | 0.5810 | 0.9542 | 0.8331 | TN 417 / FP 20 / FN 88 / TP 122 | current runtime baseline |

## BUSI External Evaluation: Best Youden Operating Point

| Model/System | AUC | Threshold | Sensitivity | Specificity | Accuracy | Confusion | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ConvNeXt-Tiny fold1 | 0.8926 | 0.5400 | 0.7667 | 0.8879 | 0.8485 | TN 388 / FP 49 / FN 49 / TP 161 | single fold |
| ConvNeXt-Tiny 5-fold ensemble | 0.9044 | 0.4400 | 0.7762 | 0.8741 | 0.8423 | TN 382 / FP 55 / FN 47 / TP 163 | new main-model candidate |
| EfficientNetV2-S 5-fold + TTA | 0.8997 | 0.3300 | 0.8000 | 0.8764 | 0.8516 | TN 383 / FP 54 / FN 42 / TP 168 | existing ensemble baseline |
| EfficientNetV2-S + DenseNet121 mixed ensemble | 0.9052 | 0.2700 | 0.8095 | 0.8719 | 0.8516 | TN 381 / FP 56 / FN 40 / TP 170 | current runtime baseline |

## Interpretation

- ConvNeXt-Tiny 5-fold improves BUSI AUC from the single-fold `0.8926` to `0.9044`.
- ConvNeXt-Tiny 5-fold is slightly above the EfficientNetV2-S 5-fold + TTA baseline by AUC: `0.9044` vs `0.8997`.
- The current mixed EfficientNetV2-S + DenseNet121 ensemble is still marginally higher by AUC: `0.9052` vs `0.9044`.
- ConvNeXt-Tiny has a strong specificity/accuracy balance, but its best-Youden sensitivity is still below the mixed ensemble.
- Because ConvNeXt-Tiny is now close to the current runtime baseline, it is a valid main-model candidate for the next ensemble round.

## Next Ensemble Direction

- Do not replace the runtime model yet; the mixed EfficientNetV2-S + DenseNet121 ensemble still has the top AUC.
- Next priority: search a low-weight Swin-Tiny contribution only after either training Swin-Tiny five folds or explicitly marking it as a single-fold exploratory member.
- More promising immediate search: ConvNeXt-Tiny 5-fold + current mixed ensemble weight search, because both sides already have five-fold evidence.
- If the goal is a clean new family ensemble, train Swin-Tiny folds 2-5 with the timm-aware recipe, then search ConvNeXt:Swin weights.

## Artifacts

- Train config: `configs/classifier/convnext_tiny_timm_recipe.yml`
- Inference config: `configs/inference/convnext_tiny_timm_recipe_5fold.yml`
- BUSI report: `artifacts/reports/busi_convnext_tiny_timm_recipe_5fold.json`
- Threshold report: `artifacts/reports/threshold_analysis_convnext_tiny_timm_recipe_5fold.md`
