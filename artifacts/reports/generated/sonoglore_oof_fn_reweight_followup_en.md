# SonoGloReNet OOF-FN Reweighting Follow-up Report

## Date
- 2026-06-19

## Objective
- Test whether a more conservative malignant-FN reweighting strategy can improve internal and external performance without further amplifying benign false positives.

## Experimental-design issue fixed first
- The earlier `fold1` weight CSV was built from `fold1` validation predictions only.
- That file had zero overlap with the `fold1` training split.
- As a result, the training config loaded a weight file, but no training sample actually received reweighting.
- This round fixed the design by generating BUSBRA-wide weights from `5-fold OOF` predictions.
- A new `sample_weight_hit_count` guard was also added to training so future zero-hit weight files fail fast.

## Reference line
- `posw17`: current strongest external SonoGloReNet single-model reference.
- fold1 internal AUC: `0.917385`
- fold1 BUSI AUC: `0.904626`
- 5-fold OOF AUC: `0.894437`

## Candidate 1: fnonly_mild
- Weight file: `artifacts/reports/generated/sonoglore_posw17_oof_sample_weights_fnonly_mild.csv`
- Total weighted samples: `133`
- Weighted samples hitting fold1 train: `109`
- fold1 internal AUC: `0.908281`, delta vs reference `-0.009104`
- fold1 BUSI AUC: `0.898916`, delta vs reference `-0.005710`
- Decision: reject

## Candidate 2: fnextreme
- Weight file: `artifacts/reports/generated/sonoglore_posw17_oof_sample_weights_fnextreme.csv`
- Total weighted samples: `100`
- Weighted samples hitting fold1 train: `84`
- fold1 internal AUC: `0.905851`, delta vs reference `-0.011534`
- fold1 BUSI AUC: `0.888215`, delta vs reference `-0.016411`
- Decision: reject

## Interpretation
- This line has now been validated with effective, non-zero training-time reweighting.
- Conservative malignant-FN reweighting did not improve internal ranking.
- It also failed to preserve external BUSI AUC against the `posw17` reference.
- That suggests the main bottleneck is not simply insufficient emphasis on malignant hard negatives.
- The remaining problem is still more consistent with benign high-confidence false positives, calibration drift, and cross-domain decision-boundary shape.

## OOF / 5-fold continuation
- Do not continue.
- Reason: neither candidate cleared the fold1 BUSI gate needed to justify broader expansion.

## Next recommendation
- Stop the FN-only reweighting line.
- Prioritize:
  1. probability calibration / logit post-processing
  2. benign-FP suppression strategies
  3. more stable external-domain generalization constraints instead of more malignant hard-case mining
