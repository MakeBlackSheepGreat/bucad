# Lesion-Scale MoE vs ConvNeXt Performance Report

Generated at: 2026-06-21 16:30:32

## Executive Summary
- The run used the local `BUCAD` Conda environment with an NVIDIA GeForce RTX 5060 Laptop GPU.
- Lesion-Scale MoE completed BUSBRA 5-fold training and BUSI external evaluation with 5-fold ensemble + crop-sweep TTA.
- On BUSI, MoE reached AUC `0.9154` versus ConvNeXt-Tiny 5-fold crop-sweep recheck AUC `0.9054`, a delta of `0.0099`.
- At the default 0.5 threshold, MoE improves specificity and F1, but lowers sensitivity. If the operating target prioritizes screening recall, a threshold-tuned or recall-oriented MoE variant is needed.

## BUSI External Evaluation

### Lesion-Scale MoE
| AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1 | TN/FP/FN/TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9154 | 0.5000 | 0.7429 | 0.9222 | 0.8640 | 0.8211 | 0.7800 | 403/34/54/156 |

Best Youden threshold: `0.3400`; Sensitivity `0.8381`, Specificity `0.8696`, F1 `0.7946`.

### ConvNeXt-Tiny 5-Fold Crop-Sweep Recheck
| AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1 | TN/FP/FN/TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9054 | 0.5000 | 0.7714 | 0.8879 | 0.8501 | 0.7678 | 0.7696 | 388/49/48/162 |

Best Youden threshold: `0.4600`; Sensitivity `0.8048`, Specificity `0.8719`, F1 `0.7770`.

### Default-Threshold Delta
| Metric | MoE | ConvNeXt | Delta |
|---|---:|---:|---:|
| AUC | 0.9154 | 0.9054 | 0.0099 |
| Sensitivity | 0.7429 | 0.7714 | -0.0286 |
| Specificity | 0.9222 | 0.8879 | 0.0343 |
| Accuracy | 0.8640 | 0.8501 | 0.0139 |
| Precision | 0.8211 | 0.7678 | 0.0533 |
| F1 | 0.7800 | 0.7696 | 0.0104 |

## BUSBRA 5-Fold Internal Validation

### Lesion-Scale MoE Best Fold Results
| Fold | Best Epoch | AUC | Sensitivity | Specificity | Accuracy | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 0.9020 | 0.7459 | 0.8893 | 0.8427 | 0.7552 |
| 2 | 9 | 0.9158 | 0.6529 | 0.9291 | 0.8400 | 0.7248 |
| 3 | 6 | 0.9504 | 0.8347 | 0.8898 | 0.8720 | 0.8080 |
| 4 | 6 | 0.9176 | 0.8279 | 0.8340 | 0.8320 | 0.7623 |
| 5 | 9 | 0.9135 | 0.7107 | 0.9252 | 0.8560 | 0.7611 |

MoE mean: AUC `0.9199 ± 0.0162`, Sensitivity `0.7544`, Specificity `0.8935`, F1 `0.7623`.

### ConvNeXt-Tiny Best Fold Results
| Fold | Best Epoch | AUC | Sensitivity | Specificity | Accuracy | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 0.9259 | 0.7213 | 0.9328 | 0.8640 | 0.7753 |
| 2 | 3 | 0.9301 | 0.9008 | 0.7913 | 0.8267 | 0.7703 |
| 3 | 30 | 0.9469 | 0.8099 | 0.9173 | 0.8827 | 0.8167 |
| 4 | 14 | 0.8875 | 0.7623 | 0.8577 | 0.8267 | 0.7410 |
| 5 | 10 | 0.9158 | 0.6777 | 0.9331 | 0.8507 | 0.7455 |

ConvNeXt mean: AUC `0.9212 ± 0.0196`, Sensitivity `0.7744`, Specificity `0.8864`, F1 `0.7698`. The older ConvNeXt training JSON files did not store F1 directly, so this table backfills F1 from the confusion matrices.

## Interpretation And Recommendation
- MoE outperforms the ConvNeXt recheck on external AUC, so lesion-scale soft routing has a measurable signal.
- At threshold 0.5, MoE is more conservative: fewer false positives and higher specificity, but more false negatives and lower sensitivity.
- Do not replace the default demo mainline yet. Next steps should be equal-expert ablation, router weight distribution analysis, and a sensitivity-oriented threshold/loss variant.

## Artifacts
- MoE training config: `configs/classifier/sonoglore_lesion_moe_convnext_tiny.yml`
- MoE inference config: `configs/inference/sonoglore_lesion_moe_convnext_tiny_5fold_tta_crop_sweep.yml`
- MoE BUSI JSON: `artifacts/reports/busi_sonoglore_lesion_moe_convnext_tiny_5fold_tta_crop_sweep.json`
- ConvNeXt inference config: `configs/inference/convnext_tiny_timm_recipe_5fold_tta_crop_sweep.yml`
- ConvNeXt BUSI recheck JSON: `artifacts/reports/busi_convnext_tiny_timm_recipe_5fold_tta_crop_sweep_recheck_20260621.json`
