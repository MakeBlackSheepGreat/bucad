<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# BUSBRA OOF Demo Mainline Error Sample Analysis

## Data Boundary

- Internal analysis uses BUSBRA OOF prediction cache: each sample is predicted by the classifier outside its corresponding fold, suitable as the internal test/error analysis perspective for the training set.
- ROI OOF cache uses BUSBRA ground-truth masks to generate ROIs, then runs the same set of classifiers; therefore it analyzes the current ROI fusion/area gating logic, and is not equivalent to the end-to-end internal performance of the segmenter's predicted masks.
- BUSI is only listed at the end of this report as a locked external validation result, and is not involved in threshold, gate, model, or sample selection.

## Internal OOF Current Demo Operating Point

- Config: `configs/inference/demo.yml`
- Threshold: `0.51`
- AUC: `0.9166`, Sensitivity: `0.8402`, Specificity: `0.8423`, Precision: `0.7183`, F1: `0.7745`
- Confusion Matrix: TN `1068`, FP `200`, FN `97`, TP `510`
- Error samples: `297`, of which FP `200`, FN `97`.

## Error Mechanisms

| Category | Count | Percentage |
| --- | ---: | ---: |
| roi_stacker_pushed_to_malignant | 120 | 40.4% |
| full_image_error_area_gate_fallback | 63 | 21.2% |
| full_image_already_benign | 59 | 19.9% |
| full_image_already_malignant | 55 | 18.5% |

### FP Mechanisms

| Category | Count | Percentage |
| --- | ---: | ---: |
| roi_stacker_pushed_to_malignant | 120 | 60.0% |
| full_image_already_malignant | 55 | 27.5% |
| full_image_error_area_gate_fallback | 25 | 12.5% |

### FN Mechanisms

| Category | Count | Percentage |
| --- | ---: | ---: |
| full_image_already_benign | 59 | 60.8% |
| full_image_error_area_gate_fallback | 38 | 39.2% |

## Confidence and Area

### Confidence Distribution

| Category | Count | Percentage |
| --- | ---: | ---: |
| confident | 218 | 73.4% |
| borderline | 54 | 18.2% |
| near_threshold | 25 | 8.4% |

### ROI Area Distribution

| Category | Count | Percentage |
| --- | ---: | ---: |
| medium | 93 | 31.3% |
| large | 77 | 25.9% |
| small | 64 | 21.5% |
| very_large | 58 | 19.5% |
| small_or_empty | 5 | 1.7% |

### Group Means

| Error Type | final mean | full mean | ROI mean | ROI area mean | GT mask area mean | image mean | image std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP | 0.6659 | 0.4892 | 0.5882 | 0.4425 | 0.0880 | 0.3254 | 0.1528 |
| FN | 0.2570 | 0.1412 | 0.2981 | 0.5530 | 0.1133 | 0.3161 | 0.1473 |

## Branch Error Attribution

| Error Type | full_eff | full_conv | roi_eff | roi_conv | full_pair | roi_pair | gate_fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP | 48.5% | 50.0% | 42.5% | 76.0% | 40.0% | 60.5% | 12.5% |
| FN | 84.5% | 92.8% | 75.3% | 73.2% | 100.0% | 77.3% | 39.2% |

## Metadata Distribution

### BIRADS

| birads | FP | FN |
| --- | ---: | ---: |
| 2 | 44 | 2 |
| 3 | 56 | 18 |
| 4 | 97 | 72 |
| 5 | 3 | 5 |

### view_side

| view_side | FP | FN |
| --- | ---: | ---: |
| left | 88 | 42 |
| right | 84 | 42 |
| single | 28 | 13 |

## Representative Internal Error Samples

### Highest Confidence FP

| sample_id | fold | p(final) | full | ROI | ROI Area | BIRADS | Mechanism |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bus_0963-l | 3 | 1.000 | 1.000 | 1.000 | 0.784 | 4 | full_image_error_area_gate_fallback |
| bus_0962-r | 4 | 0.997 | 0.997 | 0.956 | 0.771 | 4 | full_image_error_area_gate_fallback |
| bus_0270-l | 1 | 0.985 | 0.999 | 0.960 | 0.519 | 4 | full_image_already_malignant |
| bus_1038-l | 4 | 0.966 | 0.966 | 0.960 | 0.807 | 4 | full_image_error_area_gate_fallback |
| bus_1038-r | 4 | 0.954 | 0.993 | 0.885 | 0.584 | 4 | full_image_already_malignant |
| bus_0270-r | 1 | 0.953 | 0.953 | 0.916 | 0.877 | 4 | full_image_error_area_gate_fallback |
| bus_0385-r | 3 | 0.945 | 0.760 | 0.999 | 0.498 | 4 | full_image_already_malignant |
| bus_0963-r | 3 | 0.942 | 0.942 | 0.944 | 0.793 | 4 | full_image_error_area_gate_fallback |
| bus_0427-l | 2 | 0.939 | 0.982 | 0.914 | 0.553 | 5 | full_image_already_malignant |
| bus_0359-r | 3 | 0.929 | 0.829 | 0.997 | 0.664 | 4 | full_image_already_malignant |

### Highest Confidence FN

| sample_id | fold | p(final) | full | ROI | ROI Area | BIRADS | Mechanism |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bus_0298-r | 5 | 0.000 | 0.000 | 0.219 | 0.075 | 3 | full_image_error_area_gate_fallback |
| bus_0190-l | 1 | 0.001 | 0.001 | 0.600 | 0.060 | 4 | full_image_error_area_gate_fallback |
| bus_0862-l | 4 | 0.003 | 0.003 | 0.000 | 0.799 | 4 | full_image_error_area_gate_fallback |
| bus_0231-l | 3 | 0.004 | 0.000 | 0.000 | 0.689 | 4 | full_image_already_benign |
| bus_0238-r | 1 | 0.007 | 0.007 | 0.097 | 0.859 | 4 | full_image_error_area_gate_fallback |
| bus_0334-l | 4 | 0.010 | 0.010 | 0.198 | 0.902 | 3 | full_image_error_area_gate_fallback |
| bus_0287-l | 5 | 0.017 | 0.017 | 0.201 | 0.888 | 5 | full_image_error_area_gate_fallback |
| bus_0012-r | 4 | 0.018 | 0.000 | 0.019 | 0.388 | 4 | full_image_already_benign |
| bus_0052-l | 3 | 0.018 | 0.018 | 0.017 | 0.924 | 4 | full_image_error_area_gate_fallback |
| bus_0298-l | 5 | 0.019 | 0.000 | 0.016 | 0.102 | 3 | full_image_already_benign |

## BUSI Locked External Validation

- BUSI config: `configs/inference/demo.yml`
- AUC: `0.9256`, Sensitivity: `0.8667`, Specificity: `0.8467`, Precision: `0.7309`, F1: `0.7930`
- Confusion Matrix: TN `370`, FP `67`, FN `28`, TP `182`
- This external validation is only used to confirm the generalization performance of the locked configuration, and is not used for error sample selection or parameter adjustment in this round.

## Conclusion

- Under internal OOF, `120` of the FP errors are cases where ROI/stacker pushed benign samples above the threshold, and `25` are full-image false positives after area gate fallback.
- Under internal OOF, `59` of the FN errors are already biased toward benign in the full-image branch, primarily indicating classification representation/training sample coverage issues rather than purely ROI post-processing problems.
- Subsequent optimization should only base hard-case mining, augmentation, and fusion model retraining on these internal error patterns from BUSBRA OOF; after completion, perform a locked external re-check on BUSI.
