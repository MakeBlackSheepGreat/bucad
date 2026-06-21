# 当前完整 5fold 模型基准性能总表

生成时间：2026-06-21 16:36:22

## 口径说明
- 内部测试：只统计 `artifacts/reports/train_cls_*_fold1-5.json` 五折全部存在的训练组，指标取每折 `best_metrics` 后做均值和总体标准差。
- 外部测试：只统计 BUSI JSON 中明确为完整 5fold 单模型/模型族，或当前主线、集成、ROI、stacking 改进型方案的结果；排除 `single_*`、明确单折 `fold1/fold4/fold5`、通用临时 `eval*` 文件。
- F1 若旧 JSON 未直接记录，则按 confusion matrix 补算。外部表按 AUC 降序。

## 一句话结论
- 内部 5fold AUC 均值最高：SonoGloRe GRN+ECA+AsymProj posw1.6，AUC `0.9226`。
- 外部完整 5fold 单模型最高：`busi_sonoglore_lesion_moe_convnext_tiny_5fold_tta_crop_sweep.json`，AUC `0.9154`。
- 外部主线/改进型方案最高：`busi_demo_current_external.json`，AUC `0.9256`。

## 内部 BUSBRA 完整 5fold 训练模型
| Rank | 模型 | AUC均值±std | Sens | Spec | Acc | Precision | F1 | 最好/最差fold AUC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | SonoGloRe GRN+ECA+AsymProj posw1.6 | 0.9226 ± 0.0167 | 0.8008 | 0.8975 | 0.8661 | 0.7888 | 0.7946 | 0.9517/0.9018 |
| 2 | ConvNeXt-Tiny timm recipe seed123 | 0.9224 ± 0.0116 | 0.8206 | 0.8493 | 0.8400 | 0.7364 | 0.7695 | 0.9435/0.9118 |
| 3 | SonoGloRe ConvNeXt-Tiny GRN+ECA+AsymProj | 0.9223 ± 0.0123 | 0.8369 | 0.8635 | 0.8549 | 0.7513 | 0.7891 | 0.9412/0.9065 |
| 4 | ConvNeXt-Tiny timm recipe | 0.9212 ± 0.0196 | 0.7744 | 0.8864 | 0.8501 | 0.7767 | 0.7698 | 0.9469/0.8875 |
| 5 | Lesion-Scale MoE ConvNeXt-Tiny | 0.9199 ± 0.0162 | 0.7544 | 0.8935 | 0.8485 | 0.7775 | 0.7623 | 0.9504/0.9020 |
| 6 | ConvNeXtV2-Tiny timm recipe | 0.9176 ± 0.0107 | 0.7743 | 0.9069 | 0.8640 | 0.7996 | 0.7866 | 0.9280/0.8990 |
| 7 | SonoGloRe GRN+ECA+AsymProj posw1.7 | 0.9170 ± 0.0175 | 0.7859 | 0.8966 | 0.8608 | 0.7870 | 0.7856 | 0.9465/0.8934 |
| 8 | Multi-scale no-attention base | 0.9167 ± 0.0142 | 0.7511 | 0.9029 | 0.8539 | 0.7933 | 0.7685 | 0.9382/0.8955 |
| 9 | ConvNeXt-Small timm recipe | 0.9162 ± 0.0163 | 0.8007 | 0.8730 | 0.8496 | 0.7559 | 0.7749 | 0.9444/0.8936 |
| 10 | Swin-Tiny timm recipe | 0.9139 ± 0.0130 | 0.7595 | 0.8872 | 0.8459 | 0.7640 | 0.7614 | 0.9300/0.8998 |
| 11 | DenseNet121 | 0.9084 ± 0.0157 | 0.7134 | 0.9203 | 0.8533 | 0.8127 | 0.7588 | 0.9266/0.8846 |
| 12 | EfficientNetV2-S | 0.8946 ± 0.0241 | 0.6525 | 0.9054 | 0.8235 | 0.7759 | 0.7050 | 0.9248/0.8523 |

## 外部 BUSI 完整 5fold 单模型/模型族
| Rank | 结果文件 | AUC | Threshold | Sens | Spec | Acc | Precision | F1 | TN/FP/FN/TP | Youden阈值 |
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

## 外部 BUSI 主线与改进型 5fold 集成/ROI/Stacking 方案
| Rank | 结果文件 | AUC | Threshold | Sens | Spec | Acc | Precision | F1 | TN/FP/FN/TP | Youden阈值 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `busi_demo_current_external.json` | 0.9256 | 0.5100 | 0.8667 | 0.8467 | 0.8532 | 0.7309 | 0.7930 | 370/67/28/182 | 0.5100 |
| 2 | `busi_demo_mainline_baseline_2026_04_27.json` | 0.9256 | 0.5100 | 0.8667 | 0.8467 | 0.8532 | 0.7309 | 0.7930 | 370/67/28/182 | 0.5100 |
| 3 | `busi_demo_roi_area_gate_current_recheck_after_seg5fold.json` | 0.9256 | 0.5100 | 0.8667 | 0.8467 | 0.8532 | 0.7309 | 0.7930 | 370/67/28/182 | 0.5100 |
| 4 | `busi_demo_roi_area_gate_external_validation_after_busbra_oof_error_analysis.json` | 0.9256 | 0.5100 | 0.8667 | 0.8467 | 0.8532 | 0.7309 | 0.7930 | 370/67/28/182 | 0.5100 |
| 5 | `busi_demo_roi_area_gate_f1_candidate.json` | 0.9256 | 0.5100 | 0.8667 | 0.8467 | 0.8532 | 0.7309 | 0.7930 | 370/67/28/182 | 0.5100 |
| 6 | `busi_demo_roi_area_gate_mainline_eval.json` | 0.9256 | 0.5100 | 0.8667 | 0.8467 | 0.8532 | 0.7309 | 0.7930 | 370/67/28/182 | 0.5100 |
| 7 | `busi_demo_seed_diversity_oof_candidate_eval.json` | 0.9250 | 0.5000 | 0.8333 | 0.8581 | 0.8501 | 0.7384 | 0.7830 | 375/62/35/175 | 0.5000 |
| 8 | `busi_demo_roi_area_gate_oof_protocol.json` | 0.9241 | 0.5600 | 0.8429 | 0.8330 | 0.8362 | 0.7080 | 0.7696 | 364/73/33/177 | 0.5100 |
| 9 | `busi_demo_unetplusplus_resnet34_mainline_review.json` | 0.9237 | 0.5700 | 0.8333 | 0.8513 | 0.8454 | 0.7292 | 0.7778 | 372/65/35/175 | 0.5300 |
| 10 | `busi_ensemble_effnet_densenet_convnext_roi_oof_lcc_mask04.json` | 0.9229 | 0.5600 | 0.8143 | 0.8673 | 0.8501 | 0.7467 | 0.7790 | 379/58/39/171 | 0.5400 |
| 11 | `busi_demo_convnext_soup_oof_candidate_eval.json` | 0.9225 | 0.4900 | 0.8619 | 0.8238 | 0.8362 | 0.7016 | 0.7735 | 360/77/29/181 | 0.5000 |
| 12 | `busi_demo_model_zoo_oof_candidate_eval.json` | 0.9213 | 0.4100 | 0.7952 | 0.8787 | 0.8516 | 0.7591 | 0.7767 | 384/53/43/167 | 0.3700 |
| 13 | `busi_demo_roi_lcc_mainline_eval.json` | 0.9208 | 0.5500 | 0.8524 | 0.8169 | 0.8284 | 0.6911 | 0.7633 | 357/80/31/179 | 0.5100 |
| 14 | `busi_demo_roi_lcc_mainline_recheck.json` | 0.9208 | 0.5500 | 0.8524 | 0.8169 | 0.8284 | 0.6911 | 0.7633 | 357/80/31/179 | 0.5100 |
| 15 | `busi_demo_roi_oof_lcc_mask04_eval.json` | 0.9208 | 0.5500 | 0.8524 | 0.8169 | 0.8284 | 0.6911 | 0.7633 | 357/80/31/179 | 0.5100 |
| 16 | `busi_demo_roi_area_gate_seg5fold_ensemble_eval.json` | 0.9200 | 0.5100 | 0.8524 | 0.8261 | 0.8346 | 0.7020 | 0.7699 | 361/76/31/179 | 0.5500 |
| 17 | `busi_demo_roi_mainline_eval.json` | 0.9196 | 0.5500 | 0.8429 | 0.8330 | 0.8362 | 0.7080 | 0.7696 | 364/73/33/177 | 0.6500 |
| 18 | `busi_demo_roi_oof_eval.json` | 0.9196 | 0.5500 | 0.8429 | 0.8330 | 0.8362 | 0.7080 | 0.7696 | 364/73/33/177 | 0.6500 |
| 19 | `busi_demo_roi_oof_logit_blend_eval.json` | 0.9196 | 0.3700 | 0.8143 | 0.8558 | 0.8423 | 0.7308 | 0.7703 | 374/63/39/171 | 0.4800 |
| 20 | `busi_demo_roi_oof_weighted_logit_blend_eval.json` | 0.9196 | 0.3200 | 0.8476 | 0.8261 | 0.8331 | 0.7008 | 0.7672 | 361/76/32/178 | 0.4800 |
| 21 | `busi_demo_roi_area_gate_seg5fold_best_eval.json` | 0.9181 | 0.5100 | 0.8524 | 0.8398 | 0.8439 | 0.7189 | 0.7800 | 367/70/31/179 | 0.4900 |
| 22 | `busi_ensemble_effnet_densenet_convnext_auc_best_tta_cnn_identity.json` | 0.9162 | 0.5000 | 0.6952 | 0.9405 | 0.8609 | 0.8488 | 0.7644 | 411/26/64/146 | 0.4400 |
| 23 | `busi_ensemble_effnet_densenet_convnext_optimized.json` | 0.9162 | 0.4530 | 0.7476 | 0.9291 | 0.8702 | 0.8351 | 0.7889 | 406/31/53/157 | 0.4400 |
| 24 | `busi_ensemble_effnet_convnext_hflip_weight0511.json` | 0.9159 | 0.3850 | 0.7905 | 0.8947 | 0.8609 | 0.7830 | 0.7867 | 391/46/44/166 | 0.4000 |
| 25 | `busi_ensemble_effnet_densenet_convnext_auc_best_tta_convnext_hflip.json` | 0.9153 | 0.5000 | 0.6857 | 0.9382 | 0.8563 | 0.8421 | 0.7559 | 410/27/66/144 | 0.2200 |
| 26 | `busi_ensemble_effnet_convnext_crop_sweep_weight0569.json` | 0.9152 | 0.3990 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | 387/50/42/168 | 0.4000 |
| 27 | `busi_demo_current_threshold_eval.json` | 0.9151 | 0.2140 | 0.9048 | 0.7368 | 0.7913 | 0.6230 | 0.7379 | 322/115/20/190 | 0.4000 |
| 28 | `busi_ensemble_effnet_convnext_auc_best_tta_cnn_identity.json` | 0.9151 | 0.5000 | 0.7524 | 0.9176 | 0.8640 | 0.8144 | 0.7822 | 401/36/52/158 | 0.4000 |
| 29 | `busi_ensemble_effnet_convnext_optimized.json` | 0.9151 | 0.3990 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | 387/50/42/168 | 0.4000 |
| 30 | `busi_optimized_config_rerun_eval.json` | 0.9151 | 0.3990 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | 387/50/42/168 | 0.4000 |
| 31 | `busi_ensemble_effnet_densenet_convnext_auc_best_tta_convnext_rotate5.json` | 0.9151 | 0.5000 | 0.6857 | 0.9382 | 0.8563 | 0.8421 | 0.7559 | 410/27/66/144 | 0.2300 |
| 32 | `busi_ensemble_effnet_convnext_auc_best_tta_convnext_rotate5.json` | 0.9148 | 0.5000 | 0.7286 | 0.9199 | 0.8578 | 0.8138 | 0.7688 | 402/35/57/153 | 0.4600 |
| 33 | `busi_ensemble_effnet_convnext_auc_best_tta_convnext_hflip.json` | 0.9148 | 0.5000 | 0.7143 | 0.9245 | 0.8563 | 0.8197 | 0.7634 | 404/33/60/150 | 0.4000 |
| 34 | `busi_ensemble_effnet_densenet_convnext_auc_best_formal.json` | 0.9144 | 0.5000 | 0.7095 | 0.9382 | 0.8640 | 0.8466 | 0.7720 | 410/27/61/149 | 0.3500 |
| 35 | `busi_ensemble_effnet_convnext_oof_weight0656.json` | 0.9143 | 0.3270 | 0.8286 | 0.8192 | 0.8223 | 0.6877 | 0.7516 | 358/79/36/174 | 0.4200 |
| 36 | `busi_ensemble_effnet_convnext_auc_best_formal.json` | 0.9142 | 0.5000 | 0.7524 | 0.9130 | 0.8609 | 0.8061 | 0.7783 | 399/38/52/158 | 0.4700 |
| 37 | `busi_demo_roi_area_gate_pareto.json` | 0.9141 | 0.5200 | 0.8619 | 0.8398 | 0.8470 | 0.7211 | 0.7852 | 367/70/29/181 | 0.5100 |
| 38 | `busi_ensemble_effnet_convnext_hflip_weight0511_retrain_conv4_eff5.json` | 0.9140 | 0.3850 | 0.7571 | 0.8970 | 0.8516 | 0.7794 | 0.7681 | 392/45/51/159 | 0.3000 |
| 39 | `busi_demo_roi_area_gate_oof_f1.json` | 0.9092 | 0.5200 | 0.8524 | 0.8284 | 0.8362 | 0.7047 | 0.7716 | 362/75/31/179 | 0.5100 |
| 40 | `busi_demo_descriptor_router_segmenter_oof_frozen_review.json` | 0.9071 | 0.3700 | 0.8000 | 0.8627 | 0.8423 | 0.7368 | 0.7671 | 377/60/42/168 | 0.4200 |
| 41 | `busi_mixed_ensemble_weight063_eval.json` | 0.9052 | 0.5000 | 0.5810 | 0.9542 | 0.8331 | 0.8592 | 0.6932 | 417/20/88/122 | 0.2700 |
| 42 | `busi_ensemble_effnet5_densenet5_w1.0.json` | 0.9051 | 0.5000 | 0.5714 | 0.9611 | 0.8346 | 0.8759 | 0.6916 | 420/17/90/120 | 0.2300 |
| 43 | `busi_ensemble_effnet5_densenet5_w0.5.json` | 0.9050 | 0.5000 | 0.5905 | 0.9519 | 0.8346 | 0.8552 | 0.6986 | 416/21/86/124 | 0.2800 |
| 44 | `busi_mixed_ensemble_eval.json` | 0.9050 | 0.5000 | 0.5905 | 0.9519 | 0.8346 | 0.8552 | 0.6986 | 416/21/86/124 | 0.2800 |
| 45 | `busi_ensemble_effnet5_densenet5_w0.25.json` | 0.9037 | 0.5000 | 0.6286 | 0.9474 | 0.8439 | 0.8516 | 0.7233 | 414/23/78/132 | 0.2900 |
| 46 | `busi_ensemble_effnet5_densenet05.json` | 0.9021 | 0.5000 | 0.6476 | 0.9428 | 0.8470 | 0.8447 | 0.7332 | 412/25/74/136 | 0.3000 |
| 47 | `busi_ensemble_effnet5_densenet025.json` | 0.9011 | 0.5000 | 0.6476 | 0.9428 | 0.8470 | 0.8447 | 0.7332 | 412/25/74/136 | 0.3200 |
| 48 | `busi_demo_descriptor_router_frozen_review.json` | 0.8727 | 0.3600 | 0.8476 | 0.7368 | 0.7728 | 0.6075 | 0.7078 | 322/115/32/178 | 0.5500 |
| 49 | `busi_ensemble_effnet_convnext_retrain_conv4.json` | 0.8712 | 0.3990 | 0.7571 | 0.8398 | 0.8130 | 0.6943 | 0.7244 | 367/70/51/159 | 0.3600 |
| 50 | `busi_ensemble_effnet_convnext_retrain_eff5.json` | 0.8708 | 0.3990 | 0.7762 | 0.8558 | 0.8300 | 0.7212 | 0.7477 | 374/63/47/163 | 0.4300 |
| 51 | `busi_ensemble_effnet_convnext_retrain_conv4_eff5.json` | 0.8701 | 0.3990 | 0.7524 | 0.8581 | 0.8238 | 0.7182 | 0.7349 | 375/62/52/158 | 0.3600 |

## 导出文件
- 内部 CSV：`artifacts/reports/generated/all_5fold_internal_metrics.csv`
- 外部 CSV：`artifacts/reports/generated/all_5fold_external_busi_metrics.csv`
