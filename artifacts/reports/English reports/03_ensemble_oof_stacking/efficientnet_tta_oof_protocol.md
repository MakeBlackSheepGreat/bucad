# EfficientNetV2-S TTA OOF Protocol

## Boundary

- Candidate selection used BUSBRA OOF data only.
- BUSI is not read by this protocol.
- Only EfficientNetV2-S test-time augmentation and ROI stack blend are scanned; model weights are unchanged.

## Selected Candidate

- eff_view: `eff_hflip`
- roi_stack_blend_weight: `0.75`
- threshold: `0.450`

## OOF Metrics

| Scheme | Samples | Pos | Neg | AUC | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | Balanced Acc | Youden J | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current demo OOF | 1875 | 607 | 1268 | 0.9166 | 0.510 | 0.8416 | 0.8402 | 0.8423 | 0.7183 | 0.7745 | 0.8412 | 0.6825 | TN 1068 / FP 200 / FN 97 / TP 510 |
| Current demo OOF best threshold | 1875 | 607 | 1268 | 0.9166 | 0.510 | 0.8416 | 0.8402 | 0.8423 | 0.7183 | 0.7745 | 0.8412 | 0.6825 | TN 1068 / FP 200 / FN 97 / TP 510 |
| EfficientNet TTA selected OOF | 1875 | 607 | 1268 | 0.9190 | 0.450 | 0.8427 | 0.8402 | 0.8438 | 0.7203 | 0.7757 | 0.8420 | 0.6840 | TN 1070 / FP 198 / FN 97 / TP 510 |

## Top OOF Candidates

| Rank | Eff view | ROI blend | Accepted | AUC | Threshold | Sensitivity | Specificity | Precision | F1 |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | eff_hflip | 0.75 | True | 0.9190 | 0.450 | 0.8402 | 0.8438 | 0.7203 | 0.7757 |
| 2 | eff_hflip | 0.85 | False | 0.9183 | 0.460 | 0.8501 | 0.8352 | 0.7117 | 0.7748 |
| 3 | eff_crop095_100 | 0.75 | False | 0.9178 | 0.460 | 0.8468 | 0.8399 | 0.7169 | 0.7764 |
| 4 | eff_hflip | 0.95 | False | 0.9174 | 0.490 | 0.8402 | 0.8383 | 0.7133 | 0.7716 |
| 5 | eff_crop095_100 | 0.85 | False | 0.9169 | 0.490 | 0.8435 | 0.8415 | 0.7181 | 0.7758 |
| 6 | eff_hflip | 1.00 | False | 0.9169 | 0.480 | 0.8484 | 0.8265 | 0.7007 | 0.7675 |
| 7 | eff_crop090_095_100 | 0.75 | False | 0.9162 | 0.460 | 0.8468 | 0.8233 | 0.6965 | 0.7643 |
| 8 | eff_crop095_100 | 0.95 | False | 0.9160 | 0.500 | 0.8435 | 0.8312 | 0.7052 | 0.7682 |
| 9 | eff_crop090_095_100 | 0.85 | False | 0.9154 | 0.480 | 0.8468 | 0.8257 | 0.6993 | 0.7660 |
| 10 | eff_crop095_100 | 1.00 | False | 0.9153 | 0.510 | 0.8451 | 0.8289 | 0.7027 | 0.7674 |

## Decision

- Candidate config: `configs/inference/demo_efficientnet_tta_oof_candidate.yml`
- Run BUSI external validation only after treating this config as frozen.
