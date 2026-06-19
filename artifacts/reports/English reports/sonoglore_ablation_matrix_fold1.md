# SonoGloReNet Ablation Matrix Fold1 Report

> Data boundary: BUSBRA is used for training, internal validation, and model selection only. BUSI is used as the locked external evaluation set. Existing `B0 / R0 / R1` rows are reused without retraining.

Date: 2026-06-19

## Fixed References

| ID | Role | Internal AUC | BUSI AUC | BUSI Balanced Accuracy | Sensitivity | Specificity | F1-Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | baseline_reference | 0.9259 | 0.8926 | 0.8229 | 0.7762 | 0.8696 | 0.7581 |
| R0 | reference_sonoglore | 0.9096 | 0.8782 | 0.8006 | 0.8048 | 0.7963 | 0.7222 |
| R1 | appendix_failed_relu | 0.9068 | 0.5711 | 0.5173 | 0.0667 | 0.9680 | 0.1176 |

## Stage 1: Structure Ranking

| Rank | Candidate | Internal AUC | BUSI AUC | BUSI Balanced Accuracy | Sensitivity | Specificity | F1-Score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | S1 / ms_noattn | 0.9106 | 0.8809 | 0.8175 | 0.7952 | 0.8398 | 0.7472 |
| 2 | S2 / stage4_only | 0.9081 | 0.8827 | 0.8107 | 0.7952 | 0.8261 | 0.7373 |
| 3 | S3 / stage34_noattn | 0.9102 | 0.8754 | 0.7874 | 0.7762 | 0.7986 | 0.7072 |

## Stage 2: Regularization Ranking

| Rank | Candidate | Internal AUC | BUSI AUC | BUSI Balanced Accuracy | Sensitivity | Specificity | F1-Score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | S1_T0 / ms_noattn__base | 0.9106 | 0.8809 | 0.8175 | 0.7952 | 0.8398 | 0.7472 |
| 2 | S1_T3 / ms_noattn__regularized_cutmix | 0.9181 | 0.8853 | 0.8117 | 0.7905 | 0.8330 | 0.7394 |
| 3 | S1_T1 / ms_noattn__regularized | 0.9151 | 0.8709 | 0.7852 | 0.7238 | 0.8467 | 0.7086 |
| 4 | S1_T2 / ms_noattn__cutmix | 0.9119 | 0.9018 | 0.7483 | 0.9429 | 0.5538 | 0.6567 |

## Promotion Gate

- `R0` BUSI balanced accuracy: `0.8006`
- minimum allowed AUC from `R0`: `0.8732`
- candidate: `S1_T0` / `ms_noattn__base`
- candidate balanced accuracy: `0.8175`
- candidate AUC: `0.8809`
- gate decision: `promote to 5-fold`

## Final Candidate Snapshot

- candidate: `S1_T0` / `ms_noattn__base`
- config: `C:\Users\876762~1\AppData\Local\Temp\sonoglore_ablation_matrix\ms_noattn__base_fold1.yml`
- inference config: `C:\Users\876762~1\AppData\Local\Temp\sonoglore_ablation_matrix\ms_noattn__base_fold1_inference.yml`
- internal AUC: `0.9106`
- BUSI AUC: `0.8809`
- BUSI balanced accuracy: `0.8175`

## Error Analysis and Probability Shift

| Candidate | newly_worse | newly_better | both_wrong |
| --- | ---: | ---: | ---: |
| S1 / ms_noattn | 47 | 38 | 66 |
| S2 / stage4_only | 58 | 43 | 61 |
| S3 / stage34_noattn | 63 | 32 | 72 |
| S1_T0 / ms_noattn__base | 47 | 38 | 66 |
| S1_T1 / ms_noattn__regularized | 47 | 26 | 78 |
| S1_T2 / ms_noattn__cutmix | 138 | 35 | 69 |
| S1_T3 / ms_noattn__regularized_cutmix | 46 | 33 | 71 |

- baseline correct -> candidate wrong: `47`
- baseline wrong -> candidate correct: `38`
- both wrong: `66`
- benign malignant-prob mean: baseline `0.1412`, candidate `0.1920`
- malignant malignant-prob mean: baseline `0.7627`, candidate `0.7781`
- diff CSV: `C:\Users\876762330\Desktop\projects\Agent\artifacts\reports\generated\ms_noattn__base_vs_baseline_case_diff.csv`
- worsened-case CSV: `C:\Users\876762330\Desktop\projects\Agent\artifacts\reports\generated\ms_noattn__base_worsened_cases.csv`

### Representative Newly Worse Cases Top-N

- `benign (366)` / `benign`: baseline `0.0035` -> current `0.9845` (shift `0.9810`)
- `benign (360)` / `benign`: baseline `0.0132` -> current `0.9597` (shift `0.9465`)
- `benign (285)` / `benign`: baseline `0.0001` -> current `0.9328` (shift `0.9327`)
- `benign (63)` / `benign`: baseline `0.0170` -> current `0.9259` (shift `0.9089`)
- `malignant (147)` / `malignant`: baseline `0.9668` -> current `0.0687` (shift `-0.8981`)
- `benign (162)` / `benign`: baseline `0.0726` -> current `0.9707` (shift `0.8981`)
- `benign (284)` / `benign`: baseline `0.0003` -> current `0.8769` (shift `0.8767`)
- `benign (234)` / `benign`: baseline `0.0095` -> current `0.8790` (shift `0.8694`)
- `benign (159)` / `benign`: baseline `0.1069` -> current `0.9291` (shift `0.8223`)
- `malignant (161)` / `malignant`: baseline `0.9748` -> current `0.1545` (shift `-0.8204`)

## Single-Image Latency

- sample: `benign (1)`
- warm-up runs: `2`, measured runs: `10`
- `B0 baseline`: mean `0.01572` s, min `0.01475` s, max `0.01648` s
- `Final candidate`: mean `0.01796` s, min `0.01558` s, max `0.02093` s

## Conclusion

- Ranking is fixed by BUSI balanced accuracy, BUSI AUC, BUSI F1, then internal validation AUC.
- Final gate result in this round: `entered 5-fold`.
- `R1` remains an appendix-only failed reference and is excluded from promotion decisions.
