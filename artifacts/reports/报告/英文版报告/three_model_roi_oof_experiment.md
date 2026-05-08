<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Three-Model ROI and OOF Fusion Bypass Experiment

Date: 2026-04-26

## Experiment Objective

- Without modifying the mainline `demo.yml`, check whether the three-model test line can benefit from ROI cropping and OOF fusion.
- The three-model configuration uses `EfficientNetV2-S + DenseNet121 + ConvNeXt-Tiny`, with weights `0.358 / 0.244 / 0.398`.
- ROI OOF uses mask threshold `0.50`; deployment evaluation uses mask threshold `0.40` and the existing `segmenter_fold1.pt`.

## OOF Fusion

- Features: Three-model full-image probabilities + three-model ROI probabilities.
- Feature mode: `logit`.
- OOF CV AUC: `0.9288`.
- OOF recommended threshold: `0.56`.

## Results Comparison

| Approach | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-Model ROI OOF LCC Mainline | 0.9208 | 0.550 | 0.8524 | 0.8169 | 0.8284 | 0.6911 | 0.7633 |
| Three-Model Full-Image Baseline | 0.9162 | 0.453 | 0.7476 | 0.9291 | 0.8702 | 0.8351 | 0.7889 |
| Three-Model ROI OOF Stacking@OOF Threshold | 0.9210 | 0.560 | 0.8048 | 0.8810 | 0.8563 | 0.7647 | 0.7842 |
| Three-Model ROI OOF Stacking@Youden | 0.9210 | 0.550 | 0.8238 | 0.8719 | 0.8563 | 0.7555 | 0.7882 |
| Three-Model ROI OOF LCC Formal Config Re-run | 0.9229 | 0.560 | 0.8143 | 0.8673 | 0.8501 | 0.7467 | 0.7790 |
| Three-Model ROI-only@Youden | 0.9202 | 0.450 | 0.7905 | 0.9153 | 0.8748 | 0.8177 | 0.8039 |
| Three-Model Oracle ROI Stacking@Youden | 0.9199 | 0.520 | 0.8333 | 0.8696 | 0.8578 | 0.7543 | 0.7919 |

## ROI Area Distribution

| ROI Source | Fallback Count | Mean Area Ratio | Median Area Ratio | P10 | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 0 | 0.4333 | 0.3427 | 0.0511 | 0.9957 |
| segmenter | 0 | 0.7046 | 0.7969 | 0.2256 | 1.0000 |

## Conclusion

- Three-model full-image baseline: `AUC 0.9162, Sens 0.7476, Spec 0.9291, Acc 0.8702, Precision 0.8351, F1 0.7889`.
- Three-model segmenter ROI + OOF Stacking AUC: `0.9210`, a change of `+0.0048` relative to the three-model full-image baseline.
- Compared to the current two-model ROI OOF LCC mainline AUC of `0.9208`, the three-model ROI OOF changes by `+0.0002`.
- The formal config `configs/inference/ensemble_effnet_densenet_convnext_roi_oof_lcc_mask04.yml` re-run via `eval_busi.py` achieves an AUC of `0.9229`, higher than the current two-model mainline by `+0.0021`.
- This config's Precision and F1 are higher than the two-model mainline, but Sensitivity drops from `0.8524` to `0.8143`; not recommended to directly replace the mainline; more suitable as a "high AUC / more balanced false positive" test line candidate.
- If continuing the three-model direction, prioritize improving the DenseNet branch's ROI performance or switching to three-model multi-feature stacking, rather than directly replacing the current demo.
