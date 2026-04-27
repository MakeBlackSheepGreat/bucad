# Optimization Attempts 2026-04-27

## Boundary

- Candidate selection and training changes used BUSBRA internal OOF/training data only.
- BUSI was used only after the model-zoo candidate was frozen.
- No failed candidate in this report should be merged into `configs/inference/demo.yml`.

## Baseline

Current demo mainline:

- Config: `configs/inference/demo.yml`
- BUSI external AUC: `0.9256`
- BUSI Sensitivity: `0.8667`
- BUSI Specificity: `0.8467`
- BUSI Precision: `0.7309`
- BUSI F1: `0.7930`
- BUSI confusion: TN `370` / FP `67` / FN `28` / TP `182`

## Attempt 1: Model-Zoo OOF Candidate

Internal BUSBRA OOF selected:

- Config: `configs/inference/demo_model_zoo_oof_candidate.yml`
- Full-image weights:
  - EfficientNetV2-S: `0.3057`
  - ConvNeXt-Tiny: `0.0321`
  - DenseNet121: `0.2315`
  - ConvNeXt-Small: `0.0400`
  - Swin-Tiny: `0.3907`
- OOF nested AUC: `0.9352`
- OOF selected threshold: `0.41`
- OOF Sensitivity: `0.8402`
- OOF Specificity: `0.8927`
- OOF Precision: `0.7895`
- OOF F1: `0.8140`

Frozen BUSI external review:

- Report: `artifacts/reports/busi_demo_model_zoo_oof_candidate_eval.json`
- AUC: `0.9213`
- Sensitivity: `0.7952`
- Specificity: `0.8787`
- Precision: `0.7591`
- F1: `0.7767`
- Confusion: TN `384` / FP `53` / FN `43` / TP `167`

Decision: abandon. Despite a large BUSBRA OOF gain, external AUC and sensitivity fell. This is consistent with model-zoo/stacker overfitting to BUSBRA OOF distribution.

## Attempt 2: Hard-Sample Focal Retraining

Changes:

- Added `loss: focal` support to `src/engine/train_cls.py`.
- Added sample-weight support via `training.sample_weight_path`.
- Built hard-sample weights from `artifacts/reports/busbra_oof_demo_error_analysis.csv`.
- Config: `configs/classifier/convnext_tiny_hard_oof_focal.yml`

Fold1 result:

- Original ConvNeXt-Tiny fold1 best AUC: `0.9259`
- Hard/focal fold1 best AUC: `0.9051`
- Hard/focal fold1 Sensitivity: `0.7869`
- Hard/focal fold1 Specificity: `0.8221`
- Hard/focal fold1 F1: `0.7300`

Decision: stop after fold1. The setting over-focused hard cases and reduced ranking quality.

## Attempt 3: Mild Hard-Sample Retraining

Changes:

- Used cross-entropy rather than focal loss.
- Kept baseline augmentation.
- Used mild OOF-error sample weights:
  - FN: `1.5-1.8`
  - FP: `1.15-1.2`
- Config: `configs/classifier/convnext_tiny_hard_oof_mild.yml`

Fold1 result:

- Original ConvNeXt-Tiny fold1 best AUC: `0.9259`
- Mild hard-sample fold1 best AUC: `0.9086`
- Mild hard-sample fold1 Sensitivity: `0.7213`
- Mild hard-sample fold1 Specificity: `0.9051`
- Mild hard-sample fold1 F1: `0.7521`

Decision: stop after fold1. Mild weighting still underperformed the original fold1, so full 5-fold training is not justified.

## Current Conclusion

- The current demo remains the best deployable mainline.
- The biggest practical risk is overfitting BUSBRA OOF with meta-models or sample weighting that does not transfer.
- The next high-value direction should be a more conservative training experiment, such as seed/fold diversity for the already strong ConvNeXt-Tiny recipe, or a true nested OOF protocol that selects model families without reusing the same OOF predictions for hyperparameter search.

## Attempt 4: ROI Soft Gate + Multi-Scale Crop

Internal BUSBRA OOF selected:

- Report: `artifacts/reports/roi_soft_gate_multiscale_protocol.md`
- Candidate: margins `[0.2, 0.35]`, weights `[0.35, 0.65]`, min area `0.15`, max area `1.01`, ramp `0.03`, max weight `1.0`
- OOF AUC: `0.9221`
- OOF Sensitivity: `0.8402`
- OOF Specificity: `0.8691`
- OOF Precision: `0.7544`
- OOF F1: `0.7950`

Frozen BUSI external review:

- AUC: `0.9189`
- Sensitivity: `0.8429`
- Specificity: `0.8375`
- Precision: `0.7137`
- F1: `0.7729`
- Confusion: TN `366` / FP `71` / FN `33` / TP `177`

Decision: abandon. The soft gate reduced BUSBRA OOF false positives but did not transfer to BUSI.

## Attempt 5: ConvNeXt-Tiny Seed Diversity

Changes:

- Completed missing `convnext_tiny_timm_recipe_seed123` folds `1/2/3/5`.
- Added `scripts/run_seed_diversity_oof_protocol.py`.
- Selection used BUSBRA OOF only and kept the search narrow: split the existing ConvNeXt family weight between seed42 and seed123, preserving EfficientNetV2-S weight and current ROI area gate.

Internal BUSBRA OOF selected:

- Config: `configs/inference/demo_seed_diversity_oof_candidate.yml`
- Seed123 share: `0.67`
- ROI stack blend weight: `0.75`
- OOF AUC: `0.9249`
- OOF Sensitivity: `0.8402`
- OOF Specificity: `0.8667`
- OOF Precision: `0.7511`
- OOF F1: `0.7932`
- OOF confusion: TN `1099` / FP `169` / FN `97` / TP `510`

Frozen BUSI external review:

- Report: `artifacts/reports/busi_demo_seed_diversity_oof_candidate_eval.json`
- AUC: `0.9250`
- Sensitivity: `0.8333`
- Specificity: `0.8581`
- Precision: `0.7384`
- F1: `0.7830`
- Confusion: TN `375` / FP `62` / FN `35` / TP `175`

Decision: abandon. External AUC was slightly below the current demo and sensitivity dropped too much.

## Attempt 6: ConvNeXt-Tiny Weight Soup

Changes:

- Added `scripts/build_convnext_seed_soup.py`.
- Added `scripts/run_convnext_soup_oof_protocol.py`.
- Built same-fold seed42/seed123 linear weight averages with alpha `0.50`.

Internal BUSBRA OOF selected:

- Config: `configs/inference/demo_convnext_soup_oof_candidate.yml`
- ROI stack blend weight: `0.75`
- OOF AUC: `0.9232`
- OOF Sensitivity: `0.8501`
- OOF Specificity: `0.8462`
- OOF Precision: `0.7257`
- OOF F1: `0.7830`
- OOF confusion: TN `1073` / FP `195` / FN `91` / TP `516`

Frozen BUSI external review:

- Report: `artifacts/reports/busi_demo_convnext_soup_oof_candidate_eval.json`
- AUC: `0.9225`
- Sensitivity: `0.8619`
- Specificity: `0.8238`
- Precision: `0.7016`
- F1: `0.7735`
- Confusion: TN `360` / FP `77` / FN `29` / TP `181`

Decision: abandon. Weight averaging reduced runtime cost versus probability-level seed diversity, but external AUC and F1 both fell.

## Attempt 7: Light Regularization Retraining

Changes:

- Added `configs/classifier/convnext_tiny_timm_recipe_regularized.yml`.
- Tried light rotation, brightness/contrast, scale jitter, label smoothing, and higher weight decay.

Fold1 internal screen:

- Original ConvNeXt-Tiny fold1 best AUC: `0.9259`
- Regularized fold1 best AUC: `0.9096`
- Regularized fold1 Sensitivity: `0.8443`
- Regularized fold1 Specificity: `0.8656`
- Regularized fold1 F1: `0.7954`

Decision: stop after fold1. AUC was too far below the original fold1, so full 5-fold training is not justified.

## Attempt 8: CutMix Retraining

Changes:

- Added optional `training.mixup_alpha`, `training.cutmix_alpha`, and `training.mix_probability` support to `src/engine/train_cls.py`.
- Added `configs/classifier/convnext_tiny_timm_recipe_cutmix.yml`.

Fold1 internal screen:

- Original ConvNeXt-Tiny fold1 best AUC: `0.9259`
- CutMix fold1 best AUC: `0.9145`
- CutMix fold1 Sensitivity: `0.7459`
- CutMix fold1 Specificity: `0.9289`
- CutMix fold1 F1: `0.7879`

Decision: stop after fold1. CutMix improved specificity but hurt ranking and sensitivity.

## Attempt 9: EfficientNetV2-S TTA

Changes:

- Added `scripts/run_efficientnet_tta_oof_protocol.py`.
- Tested EfficientNetV2-S hflip and crop-sweep TTA while keeping classifier weights fixed.

Internal BUSBRA OOF selected:

- Config: `configs/inference/demo_efficientnet_tta_oof_candidate.yml`
- EfficientNet view: `eff_hflip`
- ROI stack blend weight: `0.75`
- OOF AUC: `0.9190`
- OOF Sensitivity: `0.8402`
- OOF Specificity: `0.8438`
- OOF Precision: `0.7203`
- OOF F1: `0.7757`

Decision: do not run BUSI. The internal gain over current demo OOF was too small for the added inference cost and does not justify spending another external validation.

## Attempt 10: ConvNeXt-Tiny 320 Input

Changes:

- Added `configs/classifier/convnext_tiny_timm_recipe_320.yml`.
- Kept the baseline ConvNeXt recipe but raised classifier input to `320`, with batch size `6` and learning rate `3e-5`.

Fold1 internal screen:

- Original ConvNeXt-Tiny fold1 best AUC: `0.9259`
- ConvNeXt-Tiny 320 fold1 best AUC: `0.9210`
- ConvNeXt-Tiny 320 fold1 Sensitivity: `0.7049`
- ConvNeXt-Tiny 320 fold1 Specificity: `0.9289`
- ConvNeXt-Tiny 320 fold1 F1: `0.7611`

Decision: stop after fold1. Higher resolution improved specificity but did not beat the original fold1 ranking quality and reduced sensitivity.

## Updated Conclusion

- None of the new frozen candidates improved the current BUSI AUC `0.9256`, so `configs/inference/demo.yml` remains unchanged.
- The strongest internal-only gain was ConvNeXt seed diversity, but it did not transfer externally; this reinforces that BUSBRA OOF is currently optimistic for extra ensemble complexity.
- Hard-sample weighting, regularization, CutMix, model-zoo stacking, seed diversity, weight soup, soft ROI gating, EfficientNet TTA, and naive 320-input training are all poor merge candidates under the current evidence.
- The next attempt should avoid more post-hoc OOF fusion and simple augmentation changes. A better next step is a lesion-aware training objective or a validation-stable pretraining/domain-adaptation strategy, screened fold-by-fold before any BUSI review.
