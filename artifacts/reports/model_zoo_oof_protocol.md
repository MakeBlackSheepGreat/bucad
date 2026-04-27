# Model-Zoo OOF Candidate Protocol

## Data Boundary

- Selection uses BUSBRA OOF predictions only.
- BUSI is not read by this script and must only be used after this candidate is frozen.
- ROI branch keeps the deployable EfficientNetV2-S + ConvNeXt-Tiny ROI pair; new model-zoo members affect the full-image branch.

## Views

| View | Role |
| --- | --- |
| eff_identity | full-image model-zoo candidate |
| conv_crop_sweep | full-image model-zoo candidate |
| densenet_identity | full-image model-zoo candidate |
| convsmall_crop_sweep | full-image model-zoo candidate |
| swin_crop_sweep | full-image model-zoo candidate |

## OOF Metrics

| Scheme | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1 | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current demo OOF | 0.9166 | 0.510 | 0.8402 | 0.8423 | 0.8416 | 0.7183 | 0.7745 | TN 1068 / FP 200 / FN 97 / TP 510 |
| selected nested OOF | 0.9352 | 0.410 | 0.8402 | 0.8927 | 0.8757 | 0.7895 | 0.8140 | TN 1132 / FP 136 / FN 97 / TP 510 |
| selected refit OOF | 0.9369 | 0.410 | 0.8369 | 0.8967 | 0.8773 | 0.7950 | 0.8154 | TN 1137 / FP 131 / FN 99 / TP 508 |

## Selected Candidate

- Name: `w_0.31_0.03_0.23_0.04_0.39_gate0.08-0.90_C1_none`
- Candidate config: `configs/inference/demo_model_zoo_oof_candidate.yml`
- Area gate: min `0.08`, max `0.90`, fallback `206` OOF samples.
- Logistic C: `1.0`, class_weight: `None`

| Full-image view | Weight |
| --- | ---: |
| eff_identity | 0.3057 |
| conv_crop_sweep | 0.0321 |
| densenet_identity | 0.2315 |
| convsmall_crop_sweep | 0.0400 |
| swin_crop_sweep | 0.3907 |

## Top Candidates

| Rank | AUC | Threshold | Sens | Spec | Precision | F1 | Gate | Weights |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.9352 | 0.410 | 0.8402 | 0.8927 | 0.7895 | 0.8140 | 0.08-0.90 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 2 | 0.9352 | 0.570 | 0.8451 | 0.8904 | 0.7868 | 0.8149 | 0.08-1.01 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 3 | 0.9351 | 0.390 | 0.8468 | 0.8841 | 0.7776 | 0.8107 | 0.08-0.90 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 4 | 0.9351 | 0.380 | 0.8402 | 0.8849 | 0.7774 | 0.8076 | 0.08-0.90 | eff_identity=0.15, conv_crop_sweep=0.14, densenet_identity=0.24, convsmall_crop_sweep=0.01, swin_crop_sweep=0.46 |
| 5 | 0.9351 | 0.390 | 0.8501 | 0.8825 | 0.7759 | 0.8113 | 0.08-1.01 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 6 | 0.9351 | 0.560 | 0.8484 | 0.8849 | 0.7791 | 0.8123 | 0.08-1.01 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 7 | 0.9351 | 0.380 | 0.8418 | 0.8856 | 0.7790 | 0.8092 | 0.08-0.90 | eff_identity=0.15, conv_crop_sweep=0.14, densenet_identity=0.24, convsmall_crop_sweep=0.01, swin_crop_sweep=0.46 |
| 8 | 0.9351 | 0.380 | 0.8550 | 0.8809 | 0.7746 | 0.8128 | 0.10-1.01 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 9 | 0.9350 | 0.550 | 0.8435 | 0.8785 | 0.7688 | 0.8044 | 0.08-1.01 | eff_identity=0.25, conv_crop_sweep=0.03, densenet_identity=0.33, convsmall_crop_sweep=0.10, swin_crop_sweep=0.30 |
| 10 | 0.9350 | 0.410 | 0.8402 | 0.8880 | 0.7822 | 0.8102 | 0.08-1.01 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |

## Decision Rule

- If the frozen BUSI external check improves AUC without materially hurting sensitivity, precision, specificity, or F1, this candidate can be considered for demo merge.
- If BUSI does not improve, keep this as an internal experiment and do not tune the candidate using BUSI feedback.
