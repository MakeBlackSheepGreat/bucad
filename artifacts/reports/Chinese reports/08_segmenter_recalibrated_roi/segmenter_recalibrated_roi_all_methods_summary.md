# Segmenter Replacement With ROI Recalibration - All Methods Summary

- Generated at: `2026-05-12T08:11:24.906389+00:00`
- Protocol: train/recalibrate only on BUSBRA; BUSI is frozen external review only.
- Mainline: `ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate`.
- Mainline BUSI AUC: `0.925580` at threshold `0.51`.
- Tested recalibrated segmenter methods: `10`.
- Candidate above mainline: `False`.

## Ranking By BUSI AUC

| Rank | Method | BUSI AUC | Delta vs Mainline | Threshold | Sensitivity | Specificity | F1 | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `unetplusplus_resnet34_bce_dice` | 0.923739 | -0.001842 | 0.57 | 0.8333 | 0.8513 | 0.7778 | `report_only` |
| 2 | `manet_resnet34_bce_dice` | 0.923183 | -0.002397 | 0.60 | 0.8143 | 0.8650 | 0.7773 | `report_only` |
| 3 | `deeplabv3plus_resnet34_bce_dice` | 0.922791 | -0.002790 | 0.60 | 0.8095 | 0.8741 | 0.7816 | `report_only` |
| 4 | `unetplusplus_densenet121_bce_dice` | 0.922671 | -0.002909 | 0.57 | 0.8238 | 0.8581 | 0.7775 | `report_only` |
| 5 | `pan_resnet34_bce_dice` | 0.922311 | -0.003269 | 0.52 | 0.8810 | 0.8078 | 0.7724 | `report_only` |
| 6 | `fpn_efficientnetb0_bce_dice` | 0.922159 | -0.003422 | 0.57 | 0.8143 | 0.8627 | 0.7755 | `report_only` |
| 7 | `fpn_densenet121_bce_dice` | 0.921712 | -0.003868 | 0.52 | 0.8952 | 0.8009 | 0.7753 | `report_only` |
| 8 | `pspnet_resnet34_bce_dice` | 0.921679 | -0.003901 | 0.57 | 0.8381 | 0.8490 | 0.7788 | `report_only` |
| 9 | `linknet_densenet121_bce_dice` | 0.921243 | -0.004337 | 0.59 | 0.8286 | 0.8604 | 0.7820 | `report_only` |
| 10 | `unetplusplus_efficientnetb0_bce_dice` | 0.919609 | -0.005971 | 0.57 | 0.8286 | 0.8444 | 0.7699 | `report_only` |

## Interpretation

- Best recalibrated method is `unetplusplus_resnet34_bce_dice` with BUSI AUC `0.923739`, still `0.001842` below the mainline.
- Recalibration is necessary: it gives each replacement segmenter its own predicted-mask OOF cache, ROI stacker and area gate. The experiment therefore tests the downstream pipeline under the new mask distribution rather than reusing an old calibration.
- Better segmentation Dice or stronger encoder does not automatically improve classification AUC. The current classifier ensemble and area gate appear tuned to the existing ROI distribution; several stronger/heavier segmenters increase sensitivity but lose specificity or global ranking quality.
- No method should be merged into `configs/inference/demo.yml` based on this run. Keep these outputs as negative/ablation evidence unless a later protocol changes the classifier-side training or ROI fusion strategy.

## Output Artifacts

- Aggregated JSON: `artifacts/reports/Chinese reports/08_segmenter_recalibrated_roi/segmenter_recalibrated_roi_all_methods_summary.json`
- External reviews: `artifacts/reports/Chinese reports/08_segmenter_recalibrated_roi/external_reviews`
- Frozen configs: `artifacts/reports/Chinese reports/08_segmenter_recalibrated_roi/frozen_configs`
- OOF prediction caches: `artifacts/reports/Chinese reports/08_segmenter_recalibrated_roi/oof_predictions`

