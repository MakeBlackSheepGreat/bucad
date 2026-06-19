# SonoGloReNet Final 5-Fold Benchmark Report

Date: 2026-06-19

## Compared Systems

- frozen baseline: `C:\Users\876762330\Desktop\projects\Agent\artifacts\reports\busi_convnext_tiny_timm_recipe_5fold.json`
- current champion: `S1_T0` / `ms_noattn__base`
- 5-fold inference config: `C:\Users\876762~1\AppData\Local\Temp\sonoglore_ablation_matrix\ms_noattn__base_5fold_inference.yml`

## BUSI 5-Fold Results

| Model | AUC | Balanced Accuracy | Sensitivity | Specificity | Accuracy | Precision | F1-Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny 5-fold baseline | 0.9044 | 0.8236 | 0.7524 | 0.8947 | 0.8485 | 0.7745 | 0.7633 |
| SonoGloReNet champion 5-fold | 0.8982 | 0.8249 | 0.7619 | 0.8879 | 0.8470 | 0.7656 | 0.7637 |

## Conclusion

- This report is generated only when the champion passes the fold1 promotion gate.
- All 5-fold metrics use the same BUSI external evaluation protocol.
