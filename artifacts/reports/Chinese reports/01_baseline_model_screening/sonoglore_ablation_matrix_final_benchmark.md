# SonoGloReNet 最终 5-fold Benchmark 报告

日期：2026-06-19

## 对照对象

- 冻结基线：`C:\Users\876762330\Desktop\projects\Agent\artifacts\reports\busi_convnext_tiny_timm_recipe_5fold.json`
- 当前冠军：`S1_T0` / `ms_noattn__base`
- 5-fold inference config：`C:\Users\876762~1\AppData\Local\Temp\sonoglore_ablation_matrix\ms_noattn__base_5fold_inference.yml`

## BUSI 5-fold 结果

| 模型 | AUC | Balanced Accuracy | Sensitivity | Specificity | Accuracy | Precision | F1-Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny 5-fold baseline | 0.9044 | 0.8236 | 0.7524 | 0.8947 | 0.8485 | 0.7745 | 0.7633 |
| SonoGloReNet champion 5-fold | 0.8982 | 0.8249 | 0.7619 | 0.8879 | 0.8470 | 0.7656 | 0.7637 |

## 结论

- 本报告仅在冠军通过 fold1 门控后生成。
- 所有 5-fold 指标均来自同一套 BUSI 外部评估口径。
