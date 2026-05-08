# ConvNeXt-Tiny Weight Soup OOF Protocol

## Boundary

- Candidate selection used BUSBRA OOF data only.
- BUSI is not read by this protocol.
- The ConvNeXt branch is replaced by same-fold seed42/seed123 linear weight averaging at alpha 0.50.

## Selected Candidate

- roi_stack_blend_weight: `0.75`
- threshold: `0.490`

## OOF Metrics

| Scheme | Samples | Pos | Neg | AUC | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | Balanced Acc | Youden J | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current demo OOF | 1875 | 607 | 1268 | 0.9166 | 0.510 | 0.8416 | 0.8402 | 0.8423 | 0.7183 | 0.7745 | 0.8412 | 0.6825 | TN 1068 / FP 200 / FN 97 / TP 510 |
| Current demo OOF best threshold | 1875 | 607 | 1268 | 0.9166 | 0.510 | 0.8416 | 0.8402 | 0.8423 | 0.7183 | 0.7745 | 0.8412 | 0.6825 | TN 1068 / FP 200 / FN 97 / TP 510 |
| ConvNeXt soup selected OOF | 1875 | 607 | 1268 | 0.9232 | 0.490 | 0.8475 | 0.8501 | 0.8462 | 0.7257 | 0.7830 | 0.8481 | 0.6963 | TN 1073 / FP 195 / FN 91 / TP 516 |

## Top OOF Candidates

| Rank | ROI blend | Accepted | AUC | Threshold | Sensitivity | Specificity | Precision | F1 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.75 | True | 0.9232 | 0.490 | 0.8501 | 0.8462 | 0.7257 | 0.7830 |
| 2 | 0.95 | True | 0.9198 | 0.530 | 0.8418 | 0.8438 | 0.7207 | 0.7766 |
| 3 | 1.00 | True | 0.9188 | 0.540 | 0.8435 | 0.8423 | 0.7191 | 0.7763 |
| 4 | 0.85 | False | 0.9217 | 0.500 | 0.8534 | 0.8383 | 0.7165 | 0.7789 |

## Decision

- Candidate config: `configs/inference/demo_convnext_soup_oof_candidate.yml`
- Run BUSI external validation only after treating this config as frozen.
