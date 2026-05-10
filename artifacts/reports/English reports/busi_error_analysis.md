<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# BUSI Demo Error Sample Analysis

## Data Boundary

- BUSI is the locked external test/evaluation set; this report can only be used for post-hoc error attribution and defense explanations.
- It is prohibited to use BUSI samples, error types, threshold distances, or ROI area distributions from this report to adjust thresholds, tune ROI gates, select segmenters, select models, train hard negatives, or decide on merging.
- Any performance optimization must be completed within BUSBRA's training/validation/OOF pipeline; BUSI is only used for external re-check after the configuration is locked.

## Current Operating Point

- Config: `configs\inference\demo.yml`
- Threshold: `0.51`
- AUC: `0.9256`, Sensitivity: `0.8667`, Specificity: `0.8467`, Precision: `0.7309`, F1: `0.7930`
- Confusion Matrix: TN `370`, FP `67`, FN `28`, TP `182`
- Error samples: `95`, of which FP `67`, FN `28`.
- Mainline probability reconstruction maximum error: `0.000089`; this error only comes from caching/floating-point differences.

## Primary Error Mechanisms

| Category | Count | Percentage |
| --- | ---: | ---: |
| full_image_error_area_gate_fallback | 32 | 33.7% |
| roi_stacker_pushed_to_malignant | 31 | 32.6% |
| full_image_already_benign | 20 | 21.1% |
| full_image_already_malignant | 12 | 12.6% |

### FP Mechanisms

| Category | Count | Percentage |
| --- | ---: | ---: |
| roi_stacker_pushed_to_malignant | 31 | 46.3% |
| full_image_error_area_gate_fallback | 24 | 35.8% |
| full_image_already_malignant | 12 | 17.9% |

### FN Mechanisms

| Category | Count | Percentage |
| --- | ---: | ---: |
| full_image_already_benign | 20 | 71.4% |
| full_image_error_area_gate_fallback | 8 | 28.6% |

## Confidence and ROI Area

### Confidence Distribution

| Category | Count | Percentage |
| --- | ---: | ---: |
| confident | 65 | 68.4% |
| borderline | 17 | 17.9% |
| near_threshold | 13 | 13.7% |

### ROI Area Distribution

| Category | Count | Percentage |
| --- | ---: | ---: |
| very_large | 31 | 32.6% |
| large | 28 | 29.5% |
| medium | 22 | 23.2% |
| small | 13 | 13.7% |
| small_or_empty | 1 | 1.1% |

### Group Means

| Error Type | final mean | full mean | ROI mean | ROI area mean | GT mask area mean | image mean | image std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP | 0.6566 | 0.5267 | 0.5921 | 0.6708 | 0.1192 | 0.3307 | 0.2048 |
| FN | 0.3318 | 0.1752 | 0.2242 | 0.4341 | 0.0718 | 0.3573 | 0.2065 |

## Branch Error Attribution

| Error Type | full_eff | full_conv | roi_eff | roi_conv | full_pair | roi_pair | gate_fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP | 35.8% | 59.7% | 50.7% | 77.6% | 53.7% | 71.6% | 35.8% |
| FN | 96.4% | 96.4% | 96.4% | 89.3% | 100.0% | 100.0% | 28.6% |

## Representative Error Samples

### Highest Confidence FP

| sample_id | p(final) | full | ROI | Area | Mechanism |
| --- | ---: | ---: | ---: | ---: | --- |
| benign (409) | 0.976 | 0.976 | 0.976 | 1.000 | full_image_error_area_gate_fallback |
| benign (397) | 0.958 | 0.958 | 0.958 | 1.000 | full_image_error_area_gate_fallback |
| benign (401) | 0.925 | 0.843 | 0.995 | 0.429 | full_image_already_malignant |
| benign (408) | 0.845 | 0.845 | 0.845 | 1.000 | full_image_error_area_gate_fallback |
| benign (42) | 0.837 | 0.830 | 0.836 | 0.679 | full_image_already_malignant |
| benign (400) | 0.835 | 0.835 | 0.835 | 1.000 | full_image_error_area_gate_fallback |
| benign (131) | 0.834 | 0.838 | 0.811 | 0.678 | full_image_already_malignant |
| benign (410) | 0.813 | 0.813 | 0.813 | 1.000 | full_image_error_area_gate_fallback |
| benign (407) | 0.812 | 0.812 | 0.812 | 1.000 | full_image_error_area_gate_fallback |
| benign (412) | 0.808 | 0.808 | 0.808 | 1.000 | full_image_error_area_gate_fallback |

### Highest Confidence FN

| sample_id | p(final) | full | ROI | Area | Mechanism |
| --- | ---: | ---: | ---: | ---: | --- |
| malignant (10) | 0.088 | 0.007 | 0.011 | 0.124 | full_image_already_benign |
| malignant (112) | 0.095 | 0.006 | 0.018 | 0.124 | full_image_already_benign |
| malignant (26) | 0.154 | 0.013 | 0.055 | 0.253 | full_image_already_benign |
| malignant (200) | 0.160 | 0.017 | 0.041 | 0.227 | full_image_already_benign |
| malignant (27) | 0.169 | 0.031 | 0.020 | 0.227 | full_image_already_benign |
| malignant (121) | 0.203 | 0.011 | 0.224 | 0.113 | full_image_already_benign |
| malignant (15) | 0.256 | 0.256 | 0.223 | 0.927 | full_image_error_area_gate_fallback |
| malignant (73) | 0.280 | 0.079 | 0.061 | 0.176 | full_image_already_benign |
| malignant (148) | 0.285 | 0.038 | 0.208 | 0.462 | full_image_already_benign |
| malignant (28) | 0.295 | 0.033 | 0.290 | 0.141 | full_image_already_benign |

## Conclusion

- FN is not simply a threshold issue: `22/28` FN samples are more than 0.08 away from the threshold, and the full/ROI branches have mostly already suppressed the malignant scores.
- FP errors mainly fall into two categories: `31` cases where ROI/stacker pushed benign samples to malignant, and `24` cases where the full image already had false positives after ROI area gate fallback.
- The five-fold segmenter achieves higher Dice but does not improve demo metrics; the reason is more likely a mismatch between ROI area distribution and fusion model calibration -- the diagnostic branch does not simply benefit from segmentation Dice.
- If optimization is to proceed, it can only be done within BUSBRA OOF: malignant FN-style augmentation, benign hard negatives, ROI area gating, and fusion model reselection; BUSI error samples must not be fed back into training or participate in configuration selection.
