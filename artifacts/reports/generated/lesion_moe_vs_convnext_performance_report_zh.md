# Lesion-Scale MoE 与 ConvNeXt 性能测试报告

生成时间：2026-06-21 16:30:32

## 结论摘要
- 本次使用本地 `BUCAD` Conda 环境，GPU 为 NVIDIA GeForce RTX 5060 Laptop GPU。
- Lesion-Scale MoE 已完成 BUSBRA 5 折训练，并在 BUSI 外部集上完成 5 折 ensemble + crop-sweep TTA 评估。
- BUSI 外部评估中，MoE AUC 为 `0.9154`，ConvNeXt-Tiny 5 折 crop-sweep recheck AUC 为 `0.9054`，MoE 提升 `0.0099`。
- 在默认阈值 0.5 下，MoE 的 Specificity 和 F1 更高，但 Sensitivity 低于 ConvNeXt；如果目标偏筛查召回，需要进一步调阈值或训练 recall-oriented MoE 版本。

## BUSI 外部测试

### Lesion-Scale MoE
| AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1 | TN/FP/FN/TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9154 | 0.5000 | 0.7429 | 0.9222 | 0.8640 | 0.8211 | 0.7800 | 403/34/54/156 |

Youden 最优阈值：`0.3400`；对应 Sensitivity `0.8381`，Specificity `0.8696`，F1 `0.7946`。

### ConvNeXt-Tiny 5 折 Crop-Sweep Recheck
| AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1 | TN/FP/FN/TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9054 | 0.5000 | 0.7714 | 0.8879 | 0.8501 | 0.7678 | 0.7696 | 388/49/48/162 |

Youden 最优阈值：`0.4600`；对应 Sensitivity `0.8048`，Specificity `0.8719`，F1 `0.7770`。

### 默认阈值差异
| Metric | MoE | ConvNeXt | Delta |
|---|---:|---:|---:|
| AUC | 0.9154 | 0.9054 | 0.0099 |
| Sensitivity | 0.7429 | 0.7714 | -0.0286 |
| Specificity | 0.9222 | 0.8879 | 0.0343 |
| Accuracy | 0.8640 | 0.8501 | 0.0139 |
| Precision | 0.8211 | 0.7678 | 0.0533 |
| F1 | 0.7800 | 0.7696 | 0.0104 |

## BUSBRA 5 折内部验证

### Lesion-Scale MoE 每折最佳结果
| Fold | Best Epoch | AUC | Sensitivity | Specificity | Accuracy | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 0.9020 | 0.7459 | 0.8893 | 0.8427 | 0.7552 |
| 2 | 9 | 0.9158 | 0.6529 | 0.9291 | 0.8400 | 0.7248 |
| 3 | 6 | 0.9504 | 0.8347 | 0.8898 | 0.8720 | 0.8080 |
| 4 | 6 | 0.9176 | 0.8279 | 0.8340 | 0.8320 | 0.7623 |
| 5 | 9 | 0.9135 | 0.7107 | 0.9252 | 0.8560 | 0.7611 |

MoE 平均：AUC `0.9199 ± 0.0162`，Sensitivity `0.7544`，Specificity `0.8935`，F1 `0.7623`。

### ConvNeXt-Tiny 每折最佳结果
| Fold | Best Epoch | AUC | Sensitivity | Specificity | Accuracy | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 0.9259 | 0.7213 | 0.9328 | 0.8640 | 0.7753 |
| 2 | 3 | 0.9301 | 0.9008 | 0.7913 | 0.8267 | 0.7703 |
| 3 | 30 | 0.9469 | 0.8099 | 0.9173 | 0.8827 | 0.8167 |
| 4 | 14 | 0.8875 | 0.7623 | 0.8577 | 0.8267 | 0.7410 |
| 5 | 10 | 0.9158 | 0.6777 | 0.9331 | 0.8507 | 0.7455 |

ConvNeXt 平均：AUC `0.9212 ± 0.0196`，Sensitivity `0.7744`，Specificity `0.8864`，F1 `0.7698`。ConvNeXt 旧训练 JSON 未直接写入 F1，本表按 confusion matrix 补算。

## 解释与建议
- MoE 的外部 AUC 高于 ConvNeXt recheck，说明病灶尺度软路由分支有真实增益信号。
- 默认阈值下 MoE 更保守，FP 更少，Specificity 更强；代价是 FN 增多，Sensitivity 下降。
- 不建议直接替换默认 demo 主线；建议下一步做 equal-expert 消融、router 权重分布分析，以及以 Sensitivity 为目标的阈值/损失版本。

## Artifacts
- MoE training config: `configs/classifier/sonoglore_lesion_moe_convnext_tiny.yml`
- MoE inference config: `configs/inference/sonoglore_lesion_moe_convnext_tiny_5fold_tta_crop_sweep.yml`
- MoE BUSI JSON: `artifacts/reports/busi_sonoglore_lesion_moe_convnext_tiny_5fold_tta_crop_sweep.json`
- ConvNeXt inference config: `configs/inference/convnext_tiny_timm_recipe_5fold_tta_crop_sweep.yml`
- ConvNeXt BUSI recheck JSON: `artifacts/reports/busi_convnext_tiny_timm_recipe_5fold_tta_crop_sweep_recheck_20260621.json`
