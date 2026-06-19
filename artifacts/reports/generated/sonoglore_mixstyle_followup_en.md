# SonoGloReNet MixStyle Follow-up Report

## Date
- 2026-06-19

## Objective
- Test whether MixStyle can improve BUSBRA -> BUSI domain generalization with a lightweight feature-statistics mixing module.

## Reference line
- `posw17`
- fold1 internal AUC: `0.917385`
- fold1 BUSI AUC: `0.904626`

## Candidate: MixStyle
- Config: `configs/classifier/sonoglore_convnext_tiny_grn_eca_asymproj_posw17_mixstyle.yml`
- Applied stages: stage2 / stage3
- fold1 internal AUC: `0.903000`
- fold1 internal sensitivity: `0.811475`
- fold1 internal specificity: `0.814229`
- fold1 BUSI AUC: `0.881601`
- fold1 BUSI sensitivity: `0.790476`
- fold1 BUSI specificity: `0.800915`
- Decision: reject

## Interpretation
- MixStyle shifts the classifier toward a more malignant-biased operating region.
- Internal and external sensitivity move upward.
- Specificity drops substantially, and BUSI AUC degrades well below the reference line.
- This suggests the current domain-shift problem is not solved by training-time feature-statistics mixing alone.

## Combined read with previous runs
- ArcMargin / SupCon push the model toward a more conservative regime.
- MixStyle pushes the model toward a more aggressive regime.
- Neither direction yields simultaneous internal and external improvement.
- The remaining problem is therefore more consistent with unstable cross-domain decision bias than simple lack of feature separation.

## Next recommendation
- Stop adding more large losses or generic domain-generalization blocks.
- Move to:
  1. learnable logit bias correction
  2. validation-constrained threshold shaping
  3. more stable probability reshaping
