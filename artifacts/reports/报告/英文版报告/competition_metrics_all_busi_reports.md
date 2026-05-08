<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Complete BUSI Evaluation Report Competition Metrics Summary

Date: 2026-04-26

## Notes

- This summary is based on recalculating from `artifacts/reports/*.json` using saved BUSI `pathology_label` and `malignant_probability`.
- This step does not retrain models, select models, or change weights; it only corrects evaluation report fields to include the competition-required AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score.
- BUSI remains the locked external evaluation set; the best Youden values in the table are used only for operating-point analysis and should not be used as a basis for training or tuning.
- The current demo mainline has separately completed a full external evaluation; see `artifacts/reports/competition_metrics_audit.md` for details.

- Refreshed BUSI JSON report count: `86`

## Fixed-Threshold Metrics

| JSON Report | Samples | AUC | Threshold | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `busi_convnext_small_timm_recipe_5fold_tta_crop_sweep.json` | 647 | 0.9040 | 0.5000 | 0.8423 | 0.8190 | 0.7288 | 0.8535 | 0.7713 | TN 373 / FP 64 / FN 38 / TP 172 |
| `busi_convnext_tiny_current_rerun.json` | 647 | 0.9054 | 0.5000 | 0.8501 | 0.7714 | 0.7678 | 0.8879 | 0.7696 | TN 388 / FP 49 / FN 48 / TP 162 |
| `busi_convnext_tiny_fold1.json` | 647 | 0.6012 | 0.5000 | 0.6754 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | TN 437 / FP 0 / FN 210 / TP 0 |
| `busi_convnext_tiny_timm_recipe_5fold.json` | 647 | 0.9044 | 0.5000 | 0.8485 | 0.7524 | 0.7745 | 0.8947 | 0.7633 | TN 391 / FP 46 / FN 52 / TP 158 |
| `busi_convnext_tiny_timm_recipe_fold1.json` | 647 | 0.8926 | 0.5000 | 0.8393 | 0.7762 | 0.7409 | 0.8696 | 0.7581 | TN 380 / FP 57 / FN 47 / TP 163 |
| `busi_convnext_tiny_tta_crop100_hflip.json` | 647 | 0.8994 | 0.5000 | 0.8315 | 0.7762 | 0.7244 | 0.8581 | 0.7494 | TN 375 / FP 62 / FN 47 / TP 163 |
| `busi_convnext_tiny_tta_crop90_hflip.json` | 647 | 0.9018 | 0.5000 | 0.8393 | 0.7524 | 0.7524 | 0.8810 | 0.7524 | TN 385 / FP 52 / FN 52 / TP 158 |
| `busi_convnext_tiny_tta_crop_sweep.json` | 647 | 0.9054 | 0.5000 | 0.8501 | 0.7714 | 0.7678 | 0.8879 | 0.7696 | TN 388 / FP 49 / FN 48 / TP 162 |
| `busi_convnext_tiny_tta_hflip.json` | 647 | 0.9041 | 0.5000 | 0.8423 | 0.7381 | 0.7673 | 0.8924 | 0.7524 | TN 390 / FP 47 / FN 55 / TP 155 |
| `busi_convnext_tiny_tta_identity.json` | 647 | 0.8991 | 0.5000 | 0.8346 | 0.7381 | 0.7488 | 0.8810 | 0.7434 | TN 385 / FP 52 / FN 55 / TP 155 |
| `busi_convnext_tiny_tta_rotate5.json` | 647 | 0.9048 | 0.5000 | 0.8454 | 0.7381 | 0.7750 | 0.8970 | 0.7561 | TN 392 / FP 45 / FN 55 / TP 155 |
| `busi_demo_current_threshold_eval.json` | 647 | 0.9151 | 0.2140 | 0.7913 | 0.9048 | 0.6230 | 0.7368 | 0.7379 | TN 322 / FP 115 / FN 20 / TP 190 |
| `busi_demo_roi_lcc_mainline_eval.json` | 647 | 0.9208 | 0.5500 | 0.8284 | 0.8524 | 0.6911 | 0.8169 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |
| `busi_demo_roi_lcc_mainline_recheck.json` | 647 | 0.9208 | 0.5500 | 0.8284 | 0.8524 | 0.6911 | 0.8169 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |
| `busi_demo_roi_mainline_eval.json` | 647 | 0.9196 | 0.5500 | 0.8362 | 0.8429 | 0.7080 | 0.8330 | 0.7696 | TN 364 / FP 73 / FN 33 / TP 177 |
| `busi_demo_roi_oof_eval.json` | 647 | 0.9196 | 0.5500 | 0.8362 | 0.8429 | 0.7080 | 0.8330 | 0.7696 | TN 364 / FP 73 / FN 33 / TP 177 |
| `busi_demo_roi_oof_lcc_mask04_eval.json` | 647 | 0.9208 | 0.5500 | 0.8284 | 0.8524 | 0.6911 | 0.8169 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |
| `busi_demo_roi_oof_logit_blend_eval.json` | 647 | 0.9196 | 0.3700 | 0.8423 | 0.8143 | 0.7308 | 0.8558 | 0.7703 | TN 374 / FP 63 / FN 39 / TP 171 |
| `busi_demo_roi_oof_weighted_logit_blend_eval.json` | 647 | 0.9196 | 0.3200 | 0.8331 | 0.8476 | 0.7008 | 0.8261 | 0.7672 | TN 361 / FP 76 / FN 32 / TP 178 |
| `busi_densenet121_5fold_tta.json` | 647 | 0.8914 | 0.5000 | 0.8223 | 0.4952 | 0.9204 | 0.9794 | 0.6440 | TN 428 / FP 9 / FN 106 / TP 104 |
| `busi_efficientnetv2_s_5fold_hflip.json` | 647 | 0.8997 | 0.5000 | 0.8485 | 0.6810 | 0.8218 | 0.9291 | 0.7448 | TN 406 / FP 31 / FN 67 / TP 143 |
| `busi_efficientnetv2_s_5fold_identity.json` | 647 | 0.8982 | 0.5000 | 0.8439 | 0.6619 | 0.8225 | 0.9314 | 0.7335 | TN 407 / FP 30 / FN 71 / TP 139 |
| `busi_efficientnetv2_s_current_rerun.json` | 647 | 0.8982 | 0.5000 | 0.8439 | 0.6619 | 0.8225 | 0.9314 | 0.7335 | TN 407 / FP 30 / FN 71 / TP 139 |
| `busi_ensemble_effnet5_densenet025.json` | 647 | 0.9011 | 0.5000 | 0.8470 | 0.6476 | 0.8447 | 0.9428 | 0.7332 | TN 412 / FP 25 / FN 74 / TP 136 |
| `busi_ensemble_effnet5_densenet05.json` | 647 | 0.9021 | 0.5000 | 0.8470 | 0.6476 | 0.8447 | 0.9428 | 0.7332 | TN 412 / FP 25 / FN 74 / TP 136 |
| `busi_ensemble_effnet5_densenet5_w0.25.json` | 647 | 0.9037 | 0.5000 | 0.8439 | 0.6286 | 0.8516 | 0.9474 | 0.7233 | TN 414 / FP 23 / FN 78 / TP 132 |
| `busi_ensemble_effnet5_densenet5_w0.5.json` | 647 | 0.9050 | 0.5000 | 0.8346 | 0.5905 | 0.8552 | 0.9519 | 0.6986 | TN 416 / FP 21 / FN 86 / TP 124 |
| `busi_ensemble_effnet5_densenet5_w1.0.json` | 647 | 0.9051 | 0.5000 | 0.8346 | 0.5714 | 0.8759 | 0.9611 | 0.6916 | TN 420 / FP 17 / FN 90 / TP 120 |
| `busi_ensemble_effnet_convnext_auc_best_formal.json` | 647 | 0.9142 | 0.5000 | 0.8609 | 0.7524 | 0.8061 | 0.9130 | 0.7783 | TN 399 / FP 38 / FN 52 / TP 158 |
| `busi_ensemble_effnet_convnext_auc_best_tta_cnn_identity.json` | 647 | 0.9151 | 0.5000 | 0.8640 | 0.7524 | 0.8144 | 0.9176 | 0.7822 | TN 401 / FP 36 / FN 52 / TP 158 |
| `busi_ensemble_effnet_convnext_auc_best_tta_convnext_hflip.json` | 647 | 0.9148 | 0.5000 | 0.8563 | 0.7143 | 0.8197 | 0.9245 | 0.7634 | TN 404 / FP 33 / FN 60 / TP 150 |
| `busi_ensemble_effnet_convnext_auc_best_tta_convnext_rotate5.json` | 647 | 0.9148 | 0.5000 | 0.8578 | 0.7286 | 0.8138 | 0.9199 | 0.7688 | TN 402 / FP 35 / FN 57 / TP 153 |
| `busi_ensemble_effnet_convnext_crop_sweep_weight0569.json` | 647 | 0.9152 | 0.3990 | 0.8578 | 0.8000 | 0.7706 | 0.8856 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |
| `busi_ensemble_effnet_convnext_hflip_weight0511.json` | 647 | 0.9159 | 0.3850 | 0.8609 | 0.7905 | 0.7830 | 0.8947 | 0.7867 | TN 391 / FP 46 / FN 44 / TP 166 |
| `busi_ensemble_effnet_convnext_hflip_weight0511_retrain_conv4_eff5.json` | 647 | 0.9140 | 0.3850 | 0.8516 | 0.7571 | 0.7794 | 0.8970 | 0.7681 | TN 392 / FP 45 / FN 51 / TP 159 |
| `busi_ensemble_effnet_convnext_oof_weight0656.json` | 647 | 0.9143 | 0.3270 | 0.8223 | 0.8286 | 0.6877 | 0.8192 | 0.7516 | TN 358 / FP 79 / FN 36 / TP 174 |
| `busi_ensemble_effnet_convnext_optimized.json` | 647 | 0.9151 | 0.3990 | 0.8578 | 0.8000 | 0.7706 | 0.8856 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |
| `busi_ensemble_effnet_convnext_retrain_conv4.json` | 647 | 0.8712 | 0.3990 | 0.8130 | 0.7571 | 0.6943 | 0.8398 | 0.7244 | TN 367 / FP 70 / FN 51 / TP 159 |
| `busi_ensemble_effnet_convnext_retrain_conv4_eff5.json` | 647 | 0.8701 | 0.3990 | 0.8238 | 0.7524 | 0.7182 | 0.8581 | 0.7349 | TN 375 / FP 62 / FN 52 / TP 158 |
| `busi_ensemble_effnet_convnext_retrain_eff5.json` | 647 | 0.8708 | 0.3990 | 0.8300 | 0.7762 | 0.7212 | 0.8558 | 0.7477 | TN 374 / FP 63 / FN 47 / TP 163 |
| `busi_ensemble_effnet_densenet_convnext_auc_best_formal.json` | 647 | 0.9144 | 0.5000 | 0.8640 | 0.7095 | 0.8466 | 0.9382 | 0.7720 | TN 410 / FP 27 / FN 61 / TP 149 |
| `busi_ensemble_effnet_densenet_convnext_auc_best_tta_cnn_identity.json` | 647 | 0.9162 | 0.5000 | 0.8609 | 0.6952 | 0.8488 | 0.9405 | 0.7644 | TN 411 / FP 26 / FN 64 / TP 146 |
| `busi_ensemble_effnet_densenet_convnext_auc_best_tta_convnext_hflip.json` | 647 | 0.9153 | 0.5000 | 0.8563 | 0.6857 | 0.8421 | 0.9382 | 0.7559 | TN 410 / FP 27 / FN 66 / TP 144 |
| `busi_ensemble_effnet_densenet_convnext_auc_best_tta_convnext_rotate5.json` | 647 | 0.9151 | 0.5000 | 0.8563 | 0.6857 | 0.8421 | 0.9382 | 0.7559 | TN 410 / FP 27 / FN 66 / TP 144 |
| `busi_ensemble_effnet_densenet_convnext_optimized.json` | 647 | 0.9162 | 0.4530 | 0.8702 | 0.7476 | 0.8351 | 0.9291 | 0.7889 | TN 406 / FP 31 / FN 53 / TP 157 |
| `busi_ensemble_effnet_densenet_fold1.json` | 647 | 0.8767 | 0.5000 | 0.8207 | 0.5524 | 0.8406 | 0.9497 | 0.6667 | TN 415 / FP 22 / FN 94 / TP 116 |
| `busi_ensemble_effnet_mobilenet_fold1.json` | 647 | 0.8629 | 0.5000 | 0.8223 | 0.7095 | 0.7340 | 0.8764 | 0.7215 | TN 383 / FP 54 / FN 61 / TP 149 |
| `busi_ensemble_effnet_resnet_fold1.json` | 647 | 0.8612 | 0.5000 | 0.8223 | 0.6857 | 0.7461 | 0.8879 | 0.7146 | TN 388 / FP 49 / FN 66 / TP 144 |
| `busi_eval.json` | 647 | 0.7564 | 0.5000 | 0.7388 | 0.6095 | 0.5953 | 0.8009 | 0.6024 | TN 350 / FP 87 / FN 82 / TP 128 |
| `busi_eval_final.json` | 647 | 0.8955 | 0.5000 | 0.8501 | 0.6667 | 0.8383 | 0.9382 | 0.7427 | TN 410 / FP 27 / FN 70 / TP 140 |
| `busi_mixed_ensemble_eval.json` | 647 | 0.9050 | 0.5000 | 0.8346 | 0.5905 | 0.8552 | 0.9519 | 0.6986 | TN 416 / FP 21 / FN 86 / TP 124 |
| `busi_mixed_ensemble_weight063_eval.json` | 647 | 0.9052 | 0.5000 | 0.8331 | 0.5810 | 0.8592 | 0.9542 | 0.6932 | TN 417 / FP 20 / FN 88 / TP 122 |
| `busi_optimized_config_rerun_eval.json` | 647 | 0.9151 | 0.3990 | 0.8578 | 0.8000 | 0.7706 | 0.8856 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |
| `busi_single_baseline_resnet18_classifier_fold1.json` | 647 | 0.7543 | 0.5000 | 0.7450 | 0.6429 | 0.6000 | 0.7941 | 0.6207 | TN 347 / FP 90 / FN 75 / TP 135 |
| `busi_single_basic_cnn_fold1.json` | 647 | 0.7327 | 0.5000 | 0.6754 | 0.0429 | 0.5000 | 0.9794 | 0.0789 | TN 428 / FP 9 / FN 201 / TP 9 |
| `busi_single_comparison_densenet121_fold1.json` | 647 | 0.8766 | 0.5000 | 0.8083 | 0.4571 | 0.9057 | 0.9771 | 0.6076 | TN 427 / FP 10 / FN 114 / TP 96 |
| `busi_single_comparison_mobilenetv3_small_100_fold1.json` | 647 | 0.8431 | 0.5000 | 0.7543 | 0.7143 | 0.6024 | 0.7735 | 0.6536 | TN 338 / FP 99 / FN 60 / TP 150 |
| `busi_single_comparison_resnet18_fold1.json` | 647 | 0.8480 | 0.5000 | 0.7991 | 0.7714 | 0.6639 | 0.8124 | 0.7137 | TN 355 / FP 82 / FN 48 / TP 162 |
| `busi_single_comparison_tf_efficientnetv2_s_fold1.json` | 647 | 0.8609 | 0.5000 | 0.7465 | 0.8333 | 0.5757 | 0.7048 | 0.6809 | TN 308 / FP 129 / FN 35 / TP 175 |
| `busi_single_comparison_vgg16_fold1.json` | 647 | 0.5000 | 0.5000 | 0.6754 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | TN 437 / FP 0 / FN 210 / TP 0 |
| `busi_single_convnext_small_timm_recipe_fold1.json` | 647 | 0.8947 | 0.5000 | 0.8284 | 0.7667 | 0.7220 | 0.8581 | 0.7436 | TN 375 / FP 62 / FN 49 / TP 161 |
| `busi_single_convnext_small_timm_recipe_fold1_tta_crop_sweep.json` | 647 | 0.8991 | 0.5000 | 0.8408 | 0.7857 | 0.7399 | 0.8673 | 0.7621 | TN 379 / FP 58 / FN 45 / TP 165 |
| `busi_single_convnext_small_timm_recipe_regularized_fold1_tta_crop_sweep.json` | 647 | 0.8759 | 0.5000 | 0.8346 | 0.7714 | 0.7330 | 0.8650 | 0.7517 | TN 378 / FP 59 / FN 48 / TP 162 |
| `busi_single_convnext_tiny_fold1.json` | 647 | 0.5996 | 0.5000 | 0.6754 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | TN 437 / FP 0 / FN 210 / TP 0 |
| `busi_single_convnext_tiny_timm_recipe_fold1.json` | 647 | 0.8943 | 0.5000 | 0.8423 | 0.7762 | 0.7477 | 0.8741 | 0.7617 | TN 382 / FP 55 / FN 47 / TP 163 |
| `busi_single_convnext_tiny_timm_recipe_fold1_tta_crop_sweep.json` | 647 | 0.8953 | 0.5000 | 0.8454 | 0.7952 | 0.7455 | 0.8696 | 0.7696 | TN 380 / FP 57 / FN 43 / TP 167 |
| `busi_single_convnext_tiny_timm_recipe_fold4_tta_crop_sweep.json` | 647 | 0.8901 | 0.5000 | 0.8253 | 0.7857 | 0.7082 | 0.8444 | 0.7449 | TN 369 / FP 68 / FN 45 / TP 165 |
| `busi_single_convnext_tiny_timm_recipe_seed123_fold4_tta_crop_sweep.json` | 647 | 0.8845 | 0.5000 | 0.8114 | 0.6905 | 0.7178 | 0.8696 | 0.7039 | TN 380 / FP 57 / FN 65 / TP 145 |
| `busi_single_efficientnetv2_s_256_aug_fold1.json` | 647 | 0.8655 | 0.5000 | 0.8053 | 0.7810 | 0.6721 | 0.8169 | 0.7225 | TN 357 / FP 80 / FN 46 / TP 164 |
| `busi_single_efficientnetv2_s_256_fold1.json` | 647 | 0.8599 | 0.5000 | 0.8207 | 0.6476 | 0.7640 | 0.9039 | 0.7010 | TN 395 / FP 42 / FN 74 / TP 136 |
| `busi_single_efficientnetv2_s_320_fold1.json` | 647 | 0.8621 | 0.5000 | 0.7264 | 0.8667 | 0.5498 | 0.6590 | 0.6728 | TN 288 / FP 149 / FN 28 / TP 182 |
| `busi_single_efficientnetv2_s_bestauc_seed123_fold5_identity.json` | 647 | 0.8527 | 0.5000 | 0.8130 | 0.5857 | 0.7834 | 0.9222 | 0.6703 | TN 403 / FP 34 / FN 87 / TP 123 |
| `busi_single_efficientnetv2_s_fold1.json` | 647 | 0.8483 | 0.5000 | 0.8099 | 0.6714 | 0.7231 | 0.8764 | 0.6963 | TN 383 / FP 54 / FN 69 / TP 141 |
| `busi_single_efficientnetv2_s_fold5_identity.json` | 647 | 0.8633 | 0.5000 | 0.8161 | 0.5619 | 0.8138 | 0.9382 | 0.6648 | TN 410 / FP 27 / FN 92 / TP 118 |
| `busi_single_efficientnetv2_s_lr1e4_seed123_fold1.json` | 647 | 0.8508 | 0.5000 | 0.8284 | 0.6667 | 0.7735 | 0.9062 | 0.7161 | TN 396 / FP 41 / FN 70 / TP 140 |
| `busi_single_efficientnetv2_s_sensitive_balanced_fold1.json` | 647 | 0.8803 | 0.5000 | 0.8022 | 0.7714 | 0.6694 | 0.8169 | 0.7168 | TN 357 / FP 80 / FN 48 / TP 162 |
| `busi_single_efficientnetv2_s_sensitive_fold1.json` | 647 | 0.8573 | 0.5000 | 0.6182 | 0.9381 | 0.4571 | 0.4645 | 0.6147 | TN 203 / FP 234 / FN 13 / TP 197 |
| `busi_single_swin_tiny_patch4_window7_224_fold1.json` | 647 | 0.8242 | 0.5000 | 0.6754 | 0.8714 | 0.5000 | 0.5812 | 0.6354 | TN 254 / FP 183 / FN 27 / TP 183 |
| `busi_single_swin_tiny_timm_recipe_fold1.json` | 647 | 0.8729 | 0.5000 | 0.8300 | 0.7048 | 0.7551 | 0.8902 | 0.7291 | TN 389 / FP 48 / FN 62 / TP 148 |
| `busi_swin_tiny_patch4_window7_224_fold1.json` | 647 | 0.8223 | 0.5000 | 0.6631 | 0.8810 | 0.4894 | 0.5584 | 0.6293 | TN 244 / FP 193 / FN 25 / TP 185 |
| `busi_swin_tiny_timm_recipe_5fold.json` | 647 | 0.8867 | 0.5000 | 0.8269 | 0.7476 | 0.7269 | 0.8650 | 0.7371 | TN 378 / FP 59 / FN 53 / TP 157 |
| `busi_swin_tiny_timm_recipe_5fold_tta_crop_sweep.json` | 647 | 0.8971 | 0.5000 | 0.8485 | 0.7762 | 0.7617 | 0.8833 | 0.7689 | TN 386 / FP 51 / FN 47 / TP 163 |
| `busi_swin_tiny_timm_recipe_fold1.json` | 647 | 0.8721 | 0.5000 | 0.8300 | 0.7048 | 0.7551 | 0.8902 | 0.7291 | TN 389 / FP 48 / FN 62 / TP 148 |
| `busi_tta_eval.json` | 647 | 0.8997 | 0.5000 | 0.8485 | 0.6619 | 0.8373 | 0.9382 | 0.7394 | TN 410 / FP 27 / FN 71 / TP 139 |
| `busi_tta_fine_threshold_eval.json` | 647 | 0.8997 | 0.5000 | 0.8485 | 0.6619 | 0.8373 | 0.9382 | 0.7394 | TN 410 / FP 27 / FN 71 / TP 139 |
| `competition_metrics_demo_eval.json` | 647 | 0.9208 | 0.5500 | 0.8284 | 0.8524 | 0.6911 | 0.8169 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |

## Best Youden Operating Points

| JSON Report | Best Threshold | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Youden J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `busi_convnext_small_timm_recipe_5fold_tta_crop_sweep.json` | 0.4700 | 0.8423 | 0.8333 | 0.7231 | 0.8467 | 0.7743 | 0.6800 |
| `busi_convnext_tiny_current_rerun.json` | 0.4600 | 0.8501 | 0.8048 | 0.7511 | 0.8719 | 0.7770 | 0.6766 |
| `busi_convnext_tiny_fold1.json` | 0.1000 | 0.3246 | 1.0000 | 0.3246 | 0.0000 | 0.4901 | 0.0000 |
| `busi_convnext_tiny_timm_recipe_5fold.json` | 0.4400 | 0.8423 | 0.7762 | 0.7477 | 0.8741 | 0.7617 | 0.6503 |
| `busi_convnext_tiny_timm_recipe_fold1.json` | 0.5400 | 0.8485 | 0.7667 | 0.7667 | 0.8879 | 0.7667 | 0.6545 |
| `busi_convnext_tiny_tta_crop100_hflip.json` | 0.8200 | 0.8686 | 0.6857 | 0.8834 | 0.9565 | 0.7721 | 0.6422 |
| `busi_convnext_tiny_tta_crop90_hflip.json` | 0.2600 | 0.8161 | 0.8619 | 0.6679 | 0.7941 | 0.7526 | 0.6560 |
| `busi_convnext_tiny_tta_crop_sweep.json` | 0.4600 | 0.8501 | 0.8048 | 0.7511 | 0.8719 | 0.7770 | 0.6766 |
| `busi_convnext_tiny_tta_hflip.json` | 0.4700 | 0.8485 | 0.7667 | 0.7667 | 0.8879 | 0.7667 | 0.6545 |
| `busi_convnext_tiny_tta_identity.json` | 0.5400 | 0.8532 | 0.7381 | 0.7949 | 0.9085 | 0.7654 | 0.6466 |
| `busi_convnext_tiny_tta_rotate5.json` | 0.4000 | 0.8408 | 0.7810 | 0.7421 | 0.8696 | 0.7610 | 0.6505 |
| `busi_demo_current_threshold_eval.json` | 0.4000 | 0.8578 | 0.8000 | 0.7706 | 0.8856 | 0.7850 | 0.6856 |
| `busi_demo_roi_lcc_mainline_eval.json` | 0.5100 | 0.8269 | 0.9000 | 0.6750 | 0.7918 | 0.7714 | 0.6918 |
| `busi_demo_roi_lcc_mainline_recheck.json` | 0.5100 | 0.8269 | 0.9000 | 0.6750 | 0.7918 | 0.7714 | 0.6918 |
| `busi_demo_roi_mainline_eval.json` | 0.6500 | 0.8717 | 0.7810 | 0.8159 | 0.9153 | 0.7981 | 0.6963 |
| `busi_demo_roi_oof_eval.json` | 0.6500 | 0.8717 | 0.7810 | 0.8159 | 0.9153 | 0.7981 | 0.6963 |
| `busi_demo_roi_oof_lcc_mask04_eval.json` | 0.5100 | 0.8269 | 0.9000 | 0.6750 | 0.7918 | 0.7714 | 0.6918 |
| `busi_demo_roi_oof_logit_blend_eval.json` | 0.4800 | 0.8717 | 0.7810 | 0.8159 | 0.9153 | 0.7981 | 0.6963 |
| `busi_demo_roi_oof_weighted_logit_blend_eval.json` | 0.4800 | 0.8717 | 0.7810 | 0.8159 | 0.9153 | 0.7981 | 0.6963 |
| `busi_densenet121_5fold_tta.json` | 0.1900 | 0.8377 | 0.7952 | 0.7293 | 0.8581 | 0.7608 | 0.6534 |
| `busi_efficientnetv2_s_5fold_hflip.json` | 0.3600 | 0.8470 | 0.7905 | 0.7511 | 0.8741 | 0.7703 | 0.6646 |
| `busi_efficientnetv2_s_5fold_identity.json` | 0.3900 | 0.8578 | 0.7762 | 0.7837 | 0.8970 | 0.7799 | 0.6732 |
| `busi_efficientnetv2_s_current_rerun.json` | 0.3900 | 0.8578 | 0.7762 | 0.7837 | 0.8970 | 0.7799 | 0.6732 |
| `busi_ensemble_effnet5_densenet025.json` | 0.3200 | 0.8516 | 0.8000 | 0.7568 | 0.8764 | 0.7778 | 0.6764 |
| `busi_ensemble_effnet5_densenet05.json` | 0.3000 | 0.8516 | 0.8095 | 0.7522 | 0.8719 | 0.7798 | 0.6814 |
| `busi_ensemble_effnet5_densenet5_w0.25.json` | 0.2900 | 0.8516 | 0.8095 | 0.7522 | 0.8719 | 0.7798 | 0.6814 |
| `busi_ensemble_effnet5_densenet5_w0.5.json` | 0.2800 | 0.8501 | 0.8095 | 0.7489 | 0.8696 | 0.7780 | 0.6791 |
| `busi_ensemble_effnet5_densenet5_w1.0.json` | 0.2300 | 0.8454 | 0.8190 | 0.7350 | 0.8581 | 0.7748 | 0.6772 |
| `busi_ensemble_effnet_convnext_auc_best_formal.json` | 0.4700 | 0.8655 | 0.7714 | 0.8060 | 0.9108 | 0.7883 | 0.6822 |
| `busi_ensemble_effnet_convnext_auc_best_tta_cnn_identity.json` | 0.4000 | 0.8578 | 0.8000 | 0.7706 | 0.8856 | 0.7850 | 0.6856 |
| `busi_ensemble_effnet_convnext_auc_best_tta_convnext_hflip.json` | 0.4000 | 0.8578 | 0.7857 | 0.7783 | 0.8924 | 0.7820 | 0.6782 |
| `busi_ensemble_effnet_convnext_auc_best_tta_convnext_rotate5.json` | 0.4600 | 0.8655 | 0.7524 | 0.8187 | 0.9199 | 0.7841 | 0.6723 |
| `busi_ensemble_effnet_convnext_crop_sweep_weight0569.json` | 0.4000 | 0.8563 | 0.7952 | 0.7696 | 0.8856 | 0.7822 | 0.6808 |
| `busi_ensemble_effnet_convnext_hflip_weight0511.json` | 0.4000 | 0.8624 | 0.7810 | 0.7923 | 0.9016 | 0.7866 | 0.6826 |
| `busi_ensemble_effnet_convnext_hflip_weight0511_retrain_conv4_eff5.json` | 0.3000 | 0.8331 | 0.8333 | 0.7056 | 0.8330 | 0.7642 | 0.6663 |
| `busi_ensemble_effnet_convnext_oof_weight0656.json` | 0.4200 | 0.8578 | 0.7952 | 0.7731 | 0.8879 | 0.7840 | 0.6831 |
| `busi_ensemble_effnet_convnext_optimized.json` | 0.4000 | 0.8578 | 0.8000 | 0.7706 | 0.8856 | 0.7850 | 0.6856 |
| `busi_ensemble_effnet_convnext_retrain_conv4.json` | 0.3600 | 0.8068 | 0.8048 | 0.6680 | 0.8078 | 0.7300 | 0.6125 |
| `busi_ensemble_effnet_convnext_retrain_conv4_eff5.json` | 0.3600 | 0.8176 | 0.7857 | 0.6933 | 0.8330 | 0.7366 | 0.6187 |
| `busi_ensemble_effnet_convnext_retrain_eff5.json` | 0.4300 | 0.8423 | 0.7524 | 0.7596 | 0.8856 | 0.7560 | 0.6380 |
| `busi_ensemble_effnet_densenet_convnext_auc_best_formal.json` | 0.3500 | 0.8516 | 0.7952 | 0.7591 | 0.8787 | 0.7767 | 0.6740 |
| `busi_ensemble_effnet_densenet_convnext_auc_best_tta_cnn_identity.json` | 0.4400 | 0.8686 | 0.7476 | 0.8307 | 0.9268 | 0.7870 | 0.6744 |
| `busi_ensemble_effnet_densenet_convnext_auc_best_tta_convnext_hflip.json` | 0.2200 | 0.8223 | 0.8810 | 0.6727 | 0.7941 | 0.7629 | 0.6750 |
| `busi_ensemble_effnet_densenet_convnext_auc_best_tta_convnext_rotate5.json` | 0.2300 | 0.8300 | 0.8667 | 0.6894 | 0.8124 | 0.7679 | 0.6790 |
| `busi_ensemble_effnet_densenet_convnext_optimized.json` | 0.4400 | 0.8686 | 0.7476 | 0.8307 | 0.9268 | 0.7870 | 0.6744 |
| `busi_ensemble_effnet_densenet_fold1.json` | 0.1500 | 0.8346 | 0.8000 | 0.7210 | 0.8513 | 0.7585 | 0.6513 |
| `busi_ensemble_effnet_mobilenet_fold1.json` | 0.3800 | 0.8145 | 0.7857 | 0.6875 | 0.8284 | 0.7333 | 0.6141 |
| `busi_ensemble_effnet_resnet_fold1.json` | 0.2900 | 0.8130 | 0.8238 | 0.6732 | 0.8078 | 0.7409 | 0.6316 |
| `busi_eval.json` | 0.5100 | 0.7573 | 0.5762 | 0.6402 | 0.8444 | 0.6065 | 0.4206 |
| `busi_eval_final.json` | 0.2500 | 0.8300 | 0.8476 | 0.6953 | 0.8215 | 0.7639 | 0.6691 |
| `busi_mixed_ensemble_eval.json` | 0.2800 | 0.8501 | 0.8095 | 0.7489 | 0.8696 | 0.7780 | 0.6791 |
| `busi_mixed_ensemble_weight063_eval.json` | 0.2700 | 0.8516 | 0.8095 | 0.7522 | 0.8719 | 0.7798 | 0.6814 |
| `busi_optimized_config_rerun_eval.json` | 0.4000 | 0.8578 | 0.8000 | 0.7706 | 0.8856 | 0.7850 | 0.6856 |
| `busi_single_baseline_resnet18_classifier_fold1.json` | 0.5000 | 0.7450 | 0.6429 | 0.6000 | 0.7941 | 0.6207 | 0.4369 |
| `busi_single_basic_cnn_fold1.json` | 0.3800 | 0.7326 | 0.6048 | 0.5853 | 0.7941 | 0.5948 | 0.3988 |
| `busi_single_comparison_densenet121_fold1.json` | 0.1000 | 0.8253 | 0.6905 | 0.7513 | 0.8902 | 0.7196 | 0.5806 |
| `busi_single_comparison_mobilenetv3_small_100_fold1.json` | 0.7200 | 0.7929 | 0.6857 | 0.6792 | 0.8444 | 0.6825 | 0.5301 |
| `busi_single_comparison_resnet18_fold1.json` | 0.6300 | 0.8114 | 0.7524 | 0.6930 | 0.8398 | 0.7215 | 0.5922 |
| `busi_single_comparison_tf_efficientnetv2_s_fold1.json` | 0.8700 | 0.8022 | 0.7714 | 0.6694 | 0.8169 | 0.7168 | 0.5884 |
| `busi_single_comparison_vgg16_fold1.json` | 0.1000 | 0.3246 | 1.0000 | 0.3246 | 0.0000 | 0.4901 | 0.0000 |
| `busi_single_convnext_small_timm_recipe_fold1.json` | 0.2000 | 0.8223 | 0.8238 | 0.6892 | 0.8215 | 0.7505 | 0.6453 |
| `busi_single_convnext_small_timm_recipe_fold1_tta_crop_sweep.json` | 0.2800 | 0.8315 | 0.8381 | 0.7012 | 0.8284 | 0.7636 | 0.6665 |
| `busi_single_convnext_small_timm_recipe_regularized_fold1_tta_crop_sweep.json` | 0.5200 | 0.8377 | 0.7714 | 0.7397 | 0.8696 | 0.7552 | 0.6410 |
| `busi_single_convnext_tiny_fold1.json` | 0.1000 | 0.3246 | 1.0000 | 0.3246 | 0.0000 | 0.4901 | 0.0000 |
| `busi_single_convnext_tiny_timm_recipe_fold1.json` | 0.4800 | 0.8454 | 0.7905 | 0.7477 | 0.8719 | 0.7685 | 0.6623 |
| `busi_single_convnext_tiny_timm_recipe_fold1_tta_crop_sweep.json` | 0.4900 | 0.8439 | 0.8000 | 0.7401 | 0.8650 | 0.7689 | 0.6650 |
| `busi_single_convnext_tiny_timm_recipe_fold4_tta_crop_sweep.json` | 0.5300 | 0.8300 | 0.7857 | 0.7174 | 0.8513 | 0.7500 | 0.6370 |
| `busi_single_convnext_tiny_timm_recipe_seed123_fold4_tta_crop_sweep.json` | 0.2500 | 0.8022 | 0.8238 | 0.6553 | 0.7918 | 0.7300 | 0.6156 |
| `busi_single_efficientnetv2_s_256_aug_fold1.json` | 0.4700 | 0.8053 | 0.7905 | 0.6694 | 0.8124 | 0.7249 | 0.6028 |
| `busi_single_efficientnetv2_s_256_fold1.json` | 0.3100 | 0.8192 | 0.7143 | 0.7246 | 0.8696 | 0.7194 | 0.5839 |
| `busi_single_efficientnetv2_s_320_fold1.json` | 0.9000 | 0.7929 | 0.8048 | 0.6450 | 0.7872 | 0.7161 | 0.5919 |
| `busi_single_efficientnetv2_s_bestauc_seed123_fold5_identity.json` | 0.1000 | 0.8099 | 0.7429 | 0.6933 | 0.8421 | 0.7172 | 0.5850 |
| `busi_single_efficientnetv2_s_fold1.json` | 0.2400 | 0.7975 | 0.7571 | 0.6653 | 0.8169 | 0.7082 | 0.5741 |
| `busi_single_efficientnetv2_s_fold5_identity.json` | 0.1200 | 0.8176 | 0.7238 | 0.7170 | 0.8627 | 0.7204 | 0.5865 |
| `busi_single_efficientnetv2_s_lr1e4_seed123_fold1.json` | 0.2700 | 0.8068 | 0.7810 | 0.6749 | 0.8192 | 0.7241 | 0.6002 |
| `busi_single_efficientnetv2_s_sensitive_balanced_fold1.json` | 0.7700 | 0.8331 | 0.7143 | 0.7576 | 0.8902 | 0.7353 | 0.6044 |
| `busi_single_efficientnetv2_s_sensitive_fold1.json` | 0.9000 | 0.7450 | 0.8286 | 0.5743 | 0.7048 | 0.6784 | 0.5334 |
| `busi_single_swin_tiny_patch4_window7_224_fold1.json` | 0.5700 | 0.7573 | 0.7571 | 0.6000 | 0.7574 | 0.6695 | 0.5146 |
| `busi_single_swin_tiny_timm_recipe_fold1.json` | 0.1600 | 0.8300 | 0.7429 | 0.7358 | 0.8719 | 0.7393 | 0.6147 |
| `busi_swin_tiny_patch4_window7_224_fold1.json` | 0.5700 | 0.7527 | 0.7619 | 0.5926 | 0.7483 | 0.6667 | 0.5102 |
| `busi_swin_tiny_timm_recipe_5fold.json` | 0.6700 | 0.8547 | 0.7190 | 0.8118 | 0.9199 | 0.7626 | 0.6390 |
| `busi_swin_tiny_timm_recipe_5fold_tta_crop_sweep.json` | 0.5000 | 0.8485 | 0.7762 | 0.7617 | 0.8833 | 0.7689 | 0.6595 |
| `busi_swin_tiny_timm_recipe_fold1.json` | 0.1800 | 0.8315 | 0.7524 | 0.7349 | 0.8696 | 0.7435 | 0.6219 |
| `busi_tta_eval.json` | 0.3300 | 0.8516 | 0.8000 | 0.7568 | 0.8764 | 0.7778 | 0.6764 |
| `busi_tta_fine_threshold_eval.json` | 0.3300 | 0.8516 | 0.8000 | 0.7568 | 0.8764 | 0.7778 | 0.6764 |
| `competition_metrics_demo_eval.json` | 0.5100 | 0.8269 | 0.9000 | 0.6750 | 0.7918 | 0.7714 | 0.6918 |
