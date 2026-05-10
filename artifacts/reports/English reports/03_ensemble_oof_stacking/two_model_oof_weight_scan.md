# Two-Model BUSBRA OOF Weight Scan

## Background

- This report scans ConvNeXt-Tiny and EfficientNetV2-S blending weights using BUSBRA out-of-fold probabilities.
- BUSI is locked external evaluation only and is not used for weight selection, threshold selection, or training.
- The purpose is to select ensemble weights from internal OOF evidence and avoid tuning on BUSI.

## Summary

- Current configuration: ConvNeXt `0.573` / EfficientNet `0.427`, OOF AUC `0.919584`.
- Best OOF configuration: ConvNeXt `0.656` / EfficientNet `0.344`, OOF AUC `0.919899`.
- OOF AUC gain: `+0.000314`.
- Best OOF operating point: threshold `0.327`, Sensitivity `0.8155`, Specificity `0.8612`, Accuracy `0.8464`.

## Top 10 OOF AUC

| ConvNeXt Weight | EfficientNet Weight | OOF AUC | OOF Threshold | Sens | Precision | F1-Score | Spec | Acc | Youden J |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.656 | 0.344 | 0.919899 | 0.327 | 0.8155 | - | - | 0.8612 | 0.8464 | 0.6767 |
| 0.657 | 0.343 | 0.919899 | 0.338 | 0.8105 | - | - | 0.8667 | 0.8485 | 0.6773 |
| 0.661 | 0.339 | 0.919896 | 0.337 | 0.8105 | - | - | 0.8667 | 0.8485 | 0.6773 |
| 0.658 | 0.342 | 0.919895 | 0.338 | 0.8105 | - | - | 0.8675 | 0.8491 | 0.6781 |
| 0.655 | 0.345 | 0.919886 | 0.327 | 0.8155 | - | - | 0.8612 | 0.8464 | 0.6767 |
| 0.662 | 0.338 | 0.919883 | 0.335 | 0.8105 | - | - | 0.8651 | 0.8475 | 0.6757 |
| 0.660 | 0.340 | 0.919882 | 0.337 | 0.8105 | - | - | 0.8667 | 0.8485 | 0.6773 |
| 0.654 | 0.346 | 0.919881 | 0.327 | 0.8155 | - | - | 0.8612 | 0.8464 | 0.6767 |
| 0.659 | 0.341 | 0.919881 | 0.339 | 0.8105 | - | - | 0.8675 | 0.8491 | 0.6781 |
| 0.652 | 0.348 | 0.919873 | 0.332 | 0.8155 | - | - | 0.8628 | 0.8475 | 0.6783 |

## Interpretation

- The OOF optimum gives most weight to ConvNeXt-Tiny, so ConvNeXt-Tiny is the stronger primary contributor in this probability space.
- EfficientNetV2-S still keeps about one third of the ensemble weight, which indicates useful complementary information.
- The AUC gain is small, but the selection procedure is cleaner because it is based on BUSBRA OOF rather than BUSI fine search.

## Recommendation

- If runtime weights are updated, prefer the OOF-selected ConvNeXt `0.656` / EfficientNet `0.344` blend.
- Keep threshold selection on BUSBRA OOF or internal validation evidence; use BUSI only for final external verification.
- If ROI/LCC or another model is added, regenerate OOF probabilities and rerun the weight scan.

## Artifacts

- OOF prediction source: `artifacts\reports\oof_two_model_predictions.json`
- Scan JSON: `artifacts/reports/two_model_oof_weight_scan.json`
