# SonoGloReNet Decision-Boundary Follow-up Report

## Date
- 2026-06-19

## Objective
- Test two decision-boundary-oriented strategies against the current BUSI benign false-positive issue:
  1. an ArcFace/CosFace-style margin-cosine head
  2. a supervised contrastive auxiliary loss

## Reference line
- `posw17`
- fold1 internal AUC: `0.917385`
- fold1 BUSI AUC: `0.904626`

## Candidate 1: ArcMargin Head
- Config: `configs/classifier/sonoglore_convnext_tiny_grn_eca_asymproj_posw17_arcmargin.yml`
- fold1 internal AUC: `0.907341`
- fold1 internal specificity: `0.924901`
- fold1 internal sensitivity: `0.721311`
- fold1 BUSI AUC: `0.867457`
- Decision: reject

## Candidate 2: SupCon Auxiliary Loss
- Config: `configs/classifier/sonoglore_convnext_tiny_grn_eca_asymproj_posw17_supcon.yml`
- fold1 internal AUC: `0.911002`
- fold1 internal specificity: `0.913043`
- fold1 internal sensitivity: `0.721311`
- fold1 BUSI AUC: `0.894917`
- Decision: reject

## Shared pattern
- Both methods made the classifier more conservative.
- Internal benign false positives decreased and specificity improved.
- Malignant recall decreased at the same time.
- Neither method preserved the reference external BUSI AUC.

## Conclusion
- Stronger decision-boundary shaping alone is not sufficient for the current SonoGloReNet bottleneck.
- The remaining problem now looks more like probability bias, cross-domain calibration drift, and unstable external decision-boundary placement.

## Next recommendation
- Prioritize:
  1. post-hoc temperature calibration
  2. learnable logit bias correction
  3. probability reshaping aimed at external-domain stability
