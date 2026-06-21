# 单模型神经网络结构 5fold 基准性能对比

生成时间：2026-06-21 17:13:00

## 过滤口径
- 只看本次会话关注的单模型神经网络结构改造和单模型训练组。
- 排除主线 demo、EffNet+ConvNeXt/DenseNet 多模型 ensemble、ROI area gate、OOF stacking、descriptor router、model zoo、seed diversity、混合权重等运行时优化方案。
- 保留同一单模型的 TTA/crop-sweep，因为它不引入其他模型权重；但报告会以结果文件名保留 TTA 口径。
- 内部表仅统计 fold1-5 五个训练报告全部存在的训练组；外部表仅统计 BUSI 上完整 5fold 单模型/模型族结果。

## 结论
- 内部 BUSBRA 5fold AUC 均值最高的单模型训练组：SonoGloRe GRN+ECA+AsymProj posw1.6，AUC `0.9226`。
- 外部 BUSI 5fold AUC 最高的单模型结构：`busi_sonoglore_lesion_moe_convnext_tiny_5fold_tta_crop_sweep.json`，AUC `0.9154`。
- Lesion-Scale MoE 在外部 BUSI 单模型表中排名第 1，AUC `0.9154`；这说明 MoE 的结构收益在外部集上比普通 ConvNeXt 单模型更明显。

## 内部 BUSBRA 完整 5fold 单模型训练组
| Rank | 单模型结构/训练组 | AUC均值±std | Sens | Spec | Acc | Precision | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | SonoGloRe GRN+ECA+AsymProj posw1.6 | 0.9226 ± 0.0167 | 0.8008 | 0.8975 | 0.8661 | 0.7888 | 0.7946 |
| 2 | ConvNeXt-Tiny timm recipe seed123 | 0.9224 ± 0.0116 | 0.8206 | 0.8493 | 0.8400 | 0.7364 | 0.7695 |
| 3 | SonoGloRe GRN+ECA+AsymProj | 0.9223 ± 0.0123 | 0.8369 | 0.8635 | 0.8549 | 0.7513 | 0.7891 |
| 4 | ConvNeXt-Tiny timm recipe | 0.9212 ± 0.0196 | 0.7744 | 0.8864 | 0.8501 | 0.7767 | 0.7698 |
| 5 | Lesion-Scale MoE ConvNeXt-Tiny | 0.9199 ± 0.0162 | 0.7544 | 0.8935 | 0.8485 | 0.7775 | 0.7623 |
| 6 | ConvNeXtV2-Tiny timm recipe | 0.9176 ± 0.0107 | 0.7743 | 0.9069 | 0.8640 | 0.7996 | 0.7866 |
| 7 | SonoGloRe GRN+ECA+AsymProj posw1.7 | 0.9170 ± 0.0175 | 0.7859 | 0.8966 | 0.8608 | 0.7870 | 0.7856 |
| 8 | Multi-scale no-attention base | 0.9167 ± 0.0142 | 0.7511 | 0.9029 | 0.8539 | 0.7933 | 0.7685 |
| 9 | ConvNeXt-Small timm recipe | 0.9162 ± 0.0163 | 0.8007 | 0.8730 | 0.8496 | 0.7559 | 0.7749 |
| 10 | Swin-Tiny timm recipe | 0.9139 ± 0.0130 | 0.7595 | 0.8872 | 0.8459 | 0.7640 | 0.7614 |
| 11 | DenseNet121 | 0.9084 ± 0.0157 | 0.7134 | 0.9203 | 0.8533 | 0.8127 | 0.7588 |
| 12 | EfficientNetV2-S | 0.8946 ± 0.0241 | 0.6525 | 0.9054 | 0.8235 | 0.7759 | 0.7050 |

## 外部 BUSI 完整 5fold 单模型/模型族结果
| Rank | 外部结果文件 | AUC | Threshold | Sens | Spec | Acc | Precision | F1 | TN/FP/FN/TP | Youden阈值 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `busi_sonoglore_lesion_moe_convnext_tiny_5fold_tta_crop_sweep.json` | 0.9154 | 0.5000 | 0.7429 | 0.9222 | 0.8640 | 0.8211 | 0.7800 | 403/34/54/156 | 0.3400 |
| 2 | `busi_convnext_tiny_current_rerun.json` | 0.9054 | 0.5000 | 0.7714 | 0.8879 | 0.8501 | 0.7678 | 0.7696 | 388/49/48/162 | 0.4600 |
| 3 | `busi_convnext_tiny_timm_recipe_5fold_tta_crop_sweep_recheck_20260621.json` | 0.9054 | 0.5000 | 0.7714 | 0.8879 | 0.8501 | 0.7678 | 0.7696 | 388/49/48/162 | 0.4600 |
| 4 | `busi_convnext_tiny_tta_crop_sweep.json` | 0.9054 | 0.5000 | 0.7714 | 0.8879 | 0.8501 | 0.7678 | 0.7696 | 388/49/48/162 | 0.4600 |
| 5 | `busi_convnext_tiny_tta_rotate5.json` | 0.9048 | 0.5000 | 0.7381 | 0.8970 | 0.8454 | 0.7750 | 0.7561 | 392/45/55/155 | 0.4000 |
| 6 | `busi_convnext_tiny_timm_recipe_5fold.json` | 0.9044 | 0.5000 | 0.7524 | 0.8947 | 0.8485 | 0.7745 | 0.7633 | 391/46/52/158 | 0.4400 |
| 7 | `busi_convnext_tiny_tta_hflip.json` | 0.9041 | 0.5000 | 0.7381 | 0.8924 | 0.8423 | 0.7673 | 0.7524 | 390/47/55/155 | 0.4700 |
| 8 | `busi_convnext_small_timm_recipe_5fold_tta_crop_sweep.json` | 0.9040 | 0.5000 | 0.8190 | 0.8535 | 0.8423 | 0.7288 | 0.7713 | 373/64/38/172 | 0.4700 |
| 9 | `busi_convnext_tiny_tta_crop90_hflip.json` | 0.9018 | 0.5000 | 0.7524 | 0.8810 | 0.8393 | 0.7524 | 0.7524 | 385/52/52/158 | 0.2600 |
| 10 | `busi_efficientnetv2_s_5fold_hflip.json` | 0.8997 | 0.5000 | 0.6810 | 0.9291 | 0.8485 | 0.8218 | 0.7448 | 406/31/67/143 | 0.3600 |
| 11 | `busi_convnext_tiny_tta_crop100_hflip.json` | 0.8994 | 0.5000 | 0.7762 | 0.8581 | 0.8315 | 0.7244 | 0.7494 | 375/62/47/163 | 0.8200 |
| 12 | `busi_convnext_tiny_tta_identity.json` | 0.8991 | 0.5000 | 0.7381 | 0.8810 | 0.8346 | 0.7488 | 0.7434 | 385/52/55/155 | 0.5400 |
| 13 | `busi_efficientnetv2_s_5fold_identity.json` | 0.8982 | 0.5000 | 0.6619 | 0.9314 | 0.8439 | 0.8225 | 0.7335 | 407/30/71/139 | 0.3900 |
| 14 | `busi_swin_tiny_timm_recipe_5fold_tta_crop_sweep.json` | 0.8971 | 0.5000 | 0.7762 | 0.8833 | 0.8485 | 0.7617 | 0.7689 | 386/51/47/163 | 0.5000 |
| 15 | `busi_densenet121_5fold_tta.json` | 0.8914 | 0.5000 | 0.4952 | 0.9794 | 0.8223 | 0.9204 | 0.6440 | 428/9/106/104 | 0.1900 |
| 16 | `busi_swin_tiny_timm_recipe_5fold.json` | 0.8867 | 0.5000 | 0.7476 | 0.8650 | 0.8269 | 0.7269 | 0.7371 | 378/59/53/157 | 0.6700 |

## 导出文件
- 内部单模型 CSV：`artifacts/reports/generated/single_model_5fold_internal_metrics.csv`
- 外部单模型 CSV：`artifacts/reports/generated/single_model_5fold_external_busi_metrics.csv`
