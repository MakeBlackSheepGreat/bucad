# ConvNeXt-Tiny Seed Diversity OOF Protocol

## Boundary

- Candidate training and selection used BUSBRA training/OOF data only.
- BUSI is not read by this protocol.
- The search is intentionally narrow: split the existing ConvNeXt-Tiny family weight between seed42 and seed123, while preserving the EfficientNetV2-S weight and current ROI area gate.

## Selected Candidate

- seed123_share: `0.67`
- roi_stack_blend_weight: `0.75`
- threshold: `0.500`

## OOF Metrics

| Scheme | Samples | Pos | Neg | AUC | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | Balanced Acc | Youden J | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current demo OOF | 1875 | 607 | 1268 | 0.9166 | 0.510 | 0.8416 | 0.8402 | 0.8423 | 0.7183 | 0.7745 | 0.8412 | 0.6825 | TN 1068 / FP 200 / FN 97 / TP 510 |
| Current demo OOF best threshold | 1875 | 607 | 1268 | 0.9166 | 0.510 | 0.8416 | 0.8402 | 0.8423 | 0.7183 | 0.7745 | 0.8412 | 0.6825 | TN 1068 / FP 200 / FN 97 / TP 510 |
| Seed diversity selected OOF | 1875 | 607 | 1268 | 0.9249 | 0.500 | 0.8581 | 0.8402 | 0.8667 | 0.7511 | 0.7932 | 0.8535 | 0.7069 | TN 1099 / FP 169 / FN 97 / TP 510 |

## Top OOF Candidates

| Rank | seed123_share | ROI blend | Accepted | AUC | Threshold | Sensitivity | Specificity | Precision | F1 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.67 | 0.75 | True | 0.9249 | 0.500 | 0.8402 | 0.8667 | 0.7511 | 0.7932 |
| 2 | 0.50 | 0.75 | True | 0.9246 | 0.480 | 0.8435 | 0.8549 | 0.7356 | 0.7859 |
| 3 | 0.40 | 0.75 | True | 0.9241 | 0.480 | 0.8451 | 0.8565 | 0.7381 | 0.7880 |
| 4 | 0.67 | 0.85 | True | 0.9238 | 0.520 | 0.8418 | 0.8644 | 0.7482 | 0.7922 |
| 5 | 0.50 | 0.85 | True | 0.9236 | 0.510 | 0.8402 | 0.8580 | 0.7391 | 0.7864 |
| 6 | 0.30 | 0.75 | True | 0.9232 | 0.480 | 0.8418 | 0.8557 | 0.7363 | 0.7855 |
| 7 | 0.40 | 0.85 | True | 0.9231 | 0.500 | 0.8418 | 0.8549 | 0.7353 | 0.7849 |
| 8 | 0.67 | 0.95 | True | 0.9224 | 0.530 | 0.8418 | 0.8517 | 0.7310 | 0.7825 |
| 9 | 0.30 | 0.85 | True | 0.9223 | 0.480 | 0.8517 | 0.8454 | 0.7251 | 0.7833 |
| 10 | 0.50 | 0.95 | True | 0.9223 | 0.530 | 0.8402 | 0.8549 | 0.7349 | 0.7840 |

## Decision

- Candidate config: `configs/inference/demo_seed_diversity_oof_candidate.yml`
- Run BUSI external validation only after treating this config as frozen.
