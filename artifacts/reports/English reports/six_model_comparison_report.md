<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# 6-Model Comprehensive Comparison Report

Generated: 2026-04-29

## 1. Model Overview

| Model | timm Name | Parameters | Checkpoint Size | Fold Completeness |
|---|---|---|---|---|
| ConvNeXt-Tiny V1 | convnext_tiny | 28M | 107MB × 5 | ✅ 5/5 |
| ConvNeXt-Tiny V2 | convnextv2_tiny | 28M | 107MB × 5 | ✅ 5/5 |
| ConvNeXt-Small | convnext_small | 50M | 189MB × 5 | ✅ 5/5 |
| EfficientNetV2-S | tf_efficientnetv2_s | 21M | 78MB × 5 | ✅ 5/5 |
| DenseNet-121 | densenet121 | 8M | 28MB × 5 | ✅ 5/5 |
| Swin-Tiny | swin_tiny_patch4_window7_224 | 28M | 106MB × 5 | ✅ 5/5 |

## 2. BUSBRA 5-Fold Internal Validation AUC

| Model | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Mean | Std Dev |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny V1 | 0.9259 | 0.9301 | 0.9469 | 0.8875 | 0.9158 | 0.9212 | 0.0196 |
| ConvNeXt-Tiny V2 | 0.9278 | 0.9189 | 0.9280 | 0.8990 | 0.9143 | 0.9176 | 0.0107 |
| ConvNeXt-Small | 0.9118 | 0.9177 | 0.9444 | 0.8936 | 0.9135 | 0.9162 | 0.0163 |
| Swin-Tiny | 0.9053 | 0.9291 | 0.9300 | 0.9053 | 0.8998 | 0.9139 | 0.0130 |
| DenseNet-121 | 0.9157 | 0.9266 | 0.9212 | 0.8961 | 0.8846 | 0.9071 | 0.0174 |
| EfficientNetV2-S | 0.9248 | 0.8988 | 0.9077 | 0.8891 | 0.8523 | 0.8946 | 0.0241 |

## 3. BUSI External Test (Identity TTA, Threshold=0.5)

| Model | AUC | Sensitivity | Specificity | Accuracy | F1 | Precision | Youden Threshold |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Small | 0.9040 | 0.8190 | 0.8535 | 0.8423 | 0.7713 | 0.7288 | 0.47 |
| ConvNeXt-Tiny V1 | 0.8991 | 0.7381 | 0.8810 | 0.8346 | 0.7434 | 0.7488 | 0.54 |
| EfficientNetV2-S | 0.8982 | 0.6619 | 0.9314 | 0.8439 | 0.7335 | 0.8225 | 0.39 |
| DenseNet-121 | 0.8909 | 0.3905 | 0.9908 | 0.7960 | 0.5541 | 0.9535 | 0.18 |
| ConvNeXt-Tiny V2 | 0.8897 | 0.7333 | 0.8970 | 0.8439 | 0.7531 | 0.7739 | 0.39 |
| Swin-Tiny | 0.8831 | 0.7476 | 0.8696 | 0.8300 | 0.7406 | 0.7336 | 0.61 |

## 4. Overall Rankings

### BUSBRA Internal Validation Ranking
1. ConvNeXt-Tiny V1 (0.9212)
2. ConvNeXt-Tiny V2 (0.9176)
3. ConvNeXt-Small (0.9162)
4. Swin-Tiny (0.9139)
5. DenseNet-121 (0.9071)
6. EfficientNetV2-S (0.8946)

### BUSI External Test Ranking
1. ConvNeXt-Small (0.9040)
2. ConvNeXt-Tiny V1 (0.8991)
3. EfficientNetV2-S (0.8982)
4. DenseNet-121 (0.8909)
5. ConvNeXt-Tiny V2 (0.8897)
6. Swin-Tiny (0.8831)

## 5. Key Findings

1. ConvNeXt-Small is the best model for external generalization: BUSI AUC=0.9040, 0.5% higher than V1, and achieves the highest Sensitivity (0.819).
2. DenseNet-121 has severely shifted threshold calibration: the Youden-optimal threshold is only 0.18, and Sensitivity at 0.50 threshold is only 0.39.
3. EfficientNetV2-S has the strongest Specificity (0.931) but the lowest Sensitivity (0.662), making it a conservative model.
4. ConvNeXt-Tiny V2 shows inconsistent internal-external performance: ranked 2nd on BUSBRA but 5th on BUSI, raising concerns about generalization.
5. Swin-Tiny is consistently weak across both datasets: ranked near the bottom on both.

## 6. New Mainline Integration Recommendations

| Plan | Model Combination | Expected Advantage |
|---|---|---|
| Plan A (Recommended) | ConvNeXt-Small + ConvNeXt-Tiny V1 + EfficientNetV2-S | Best external generalization + highest internal + complementary specificity |
| Plan B | ConvNeXt-Small + ConvNeXt-Tiny V1 + ConvNeXt-Tiny V2 | Three ConvNeXt variants, good stability |
| Plan C | All 6 models | Maximum heterogeneity, but requires re-searching weights |

Recommended exclusion: DenseNet-121 (severe threshold shift) and Swin-Tiny (consistently weak across datasets)

## 7. Checkpoint File Paths

### ConvNeXt-Tiny V1
- artifacts/checkpoints/convnext_tiny_timm_recipe_fold1.pt
- artifacts/checkpoints/convnext_tiny_timm_recipe_fold2.pt
- artifacts/checkpoints/convnext_tiny_timm_recipe_fold3.pt
- artifacts/checkpoints/convnext_tiny_timm_recipe_fold4.pt
- artifacts/checkpoints/convnext_tiny_timm_recipe_fold5.pt

### ConvNeXt-Tiny V2
- artifacts/checkpoints/convnextv2_tiny_timm_recipe_fold1.pt
- artifacts/checkpoints/convnextv2_tiny_timm_recipe_fold2.pt
- artifacts/checkpoints/convnextv2_tiny_timm_recipe_fold3.pt
- artifacts/checkpoints/convnextv2_tiny_timm_recipe_fold4.pt
- artifacts/checkpoints/convnextv2_tiny_timm_recipe_fold5.pt

### ConvNeXt-Small
- artifacts/checkpoints/convnext_small_timm_recipe_fold1.pt
- artifacts/checkpoints/convnext_small_timm_recipe_fold2.pt
- artifacts/checkpoints/convnext_small_timm_recipe_fold3.pt
- artifacts/checkpoints/convnext_small_timm_recipe_fold4.pt
- artifacts/checkpoints/convnext_small_timm_recipe_fold5.pt

### EfficientNetV2-S
- artifacts/checkpoints/efficientnetv2_s_fold1.pt
- artifacts/checkpoints/efficientnetv2_s_fold2.pt
- artifacts/checkpoints/efficientnetv2_s_fold3.pt
- artifacts/checkpoints/efficientnetv2_s_fold4.pt
- artifacts/checkpoints/efficientnetv2_s_fold5.pt

### DenseNet-121
- artifacts/checkpoints/densenet121_fold1.pt
- artifacts/checkpoints/densenet121_fold2.pt
- artifacts/checkpoints/densenet121_fold3.pt
- artifacts/checkpoints/densenet121_fold4.pt
- artifacts/checkpoints/densenet121_fold5.pt

### Swin-Tiny
- artifacts/checkpoints/swin_tiny_patch4_window7_224_timm_recipe_fold1.pt
- artifacts/checkpoints/swin_tiny_patch4_window7_224_timm_recipe_fold2.pt
- artifacts/checkpoints/swin_tiny_patch4_window7_224_timm_recipe_fold3.pt
- artifacts/checkpoints/swin_tiny_patch4_window7_224_timm_recipe_fold4.pt
- artifacts/checkpoints/swin_tiny_patch4_window7_224_timm_recipe_fold5.pt

## 8. Inference Configuration File Paths

- configs/inference/convnext_tiny_timm_recipe_5fold_tta_identity.yml
- configs/inference/convnextv2_tiny_timm_recipe_5fold_tta_identity.yml
- configs/inference/convnext_small_timm_recipe_5fold_tta_crop_sweep.yml
- configs/inference/efficientnetv2_s_5fold_identity.yml
- configs/inference/densenet121_5fold_tta_identity.yml
- configs/inference/swin_tiny_timm_recipe_5fold_tta_identity.yml

## 9. Raw Evaluation Report Paths

- artifacts/reports/busi_eval_convnext_v1.json
- artifacts/reports/busi_eval_convnextv2_v2.json
- artifacts/reports/busi_eval_convnext_small_identity.json
- artifacts/reports/busi_eval_efficientnetv2_s_identity.json
- artifacts/reports/busi_eval_densenet121_identity.json
- artifacts/reports/busi_eval_swin_tiny_identity.json
