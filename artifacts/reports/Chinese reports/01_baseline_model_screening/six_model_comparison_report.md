# 6 模型全面对比报告

生成时间: 2026-04-29

## 一、模型概览

| 模型 | timm 名称 | 参数量 | Checkpoint 大小 | Fold 完整性 |
|---|---|---|---|---|
| ConvNeXt-Tiny V1 | convnext_tiny | 28M | 107MB × 5 | ✅ 5/5 |
| ConvNeXt-Tiny V2 | convnextv2_tiny | 28M | 107MB × 5 | ✅ 5/5 |
| ConvNeXt-Small | convnext_small | 50M | 189MB × 5 | ✅ 5/5 |
| EfficientNetV2-S | tf_efficientnetv2_s | 21M | 78MB × 5 | ✅ 5/5 |
| DenseNet-121 | densenet121 | 8M | 28MB × 5 | ✅ 5/5 |
| Swin-Tiny | swin_tiny_patch4_window7_224 | 28M | 106MB × 5 | ✅ 5/5 |

## 二、BUSBRA 5-Fold 内部验证 AUC

| 模型 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | 均值 | 标准差 |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny V1 | 0.9259 | 0.9301 | 0.9469 | 0.8875 | 0.9158 | 0.9212 | 0.0196 |
| ConvNeXt-Tiny V2 | 0.9278 | 0.9189 | 0.9280 | 0.8990 | 0.9143 | 0.9176 | 0.0107 |
| ConvNeXt-Small | 0.9118 | 0.9177 | 0.9444 | 0.8936 | 0.9135 | 0.9162 | 0.0163 |
| Swin-Tiny | 0.9053 | 0.9291 | 0.9300 | 0.9053 | 0.8998 | 0.9139 | 0.0130 |
| DenseNet-121 | 0.9157 | 0.9266 | 0.9212 | 0.8961 | 0.8846 | 0.9071 | 0.0174 |
| EfficientNetV2-S | 0.9248 | 0.8988 | 0.9077 | 0.8891 | 0.8523 | 0.8946 | 0.0241 |

## 三、BUSI 外部测试（Identity TTA，阈值=0.5）

| 模型 | AUC | Sensitivity | Specificity | Accuracy | F1 | Precision | Youden 阈值 |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Small | 0.9040 | 0.8190 | 0.8535 | 0.8423 | 0.7713 | 0.7288 | 0.47 |
| ConvNeXt-Tiny V1 | 0.8991 | 0.7381 | 0.8810 | 0.8346 | 0.7434 | 0.7488 | 0.54 |
| EfficientNetV2-S | 0.8982 | 0.6619 | 0.9314 | 0.8439 | 0.7335 | 0.8225 | 0.39 |
| DenseNet-121 | 0.8909 | 0.3905 | 0.9908 | 0.7960 | 0.5541 | 0.9535 | 0.18 |
| ConvNeXt-Tiny V2 | 0.8897 | 0.7333 | 0.8970 | 0.8439 | 0.7531 | 0.7739 | 0.39 |
| Swin-Tiny | 0.8831 | 0.7476 | 0.8696 | 0.8300 | 0.7406 | 0.7336 | 0.61 |

## 四、综合排名

### BUSBRA 内部验证排名
1. ConvNeXt-Tiny V1 (0.9212)
2. ConvNeXt-Tiny V2 (0.9176)
3. ConvNeXt-Small (0.9162)
4. Swin-Tiny (0.9139)
5. DenseNet-121 (0.9071)
6. EfficientNetV2-S (0.8946)

### BUSI 外部测试排名
1. ConvNeXt-Small (0.9040)
2. ConvNeXt-Tiny V1 (0.8991)
3. EfficientNetV2-S (0.8982)
4. DenseNet-121 (0.8909)
5. ConvNeXt-Tiny V2 (0.8897)
6. Swin-Tiny (0.8831)

## 五、关键发现

1. ConvNeXt-Small 是外部泛化最佳模型：BUSI AUC=0.9040，比 V1 高 0.5%，且 Sensitivity 最高（0.819）
2. DenseNet-121 阈值校准严重偏移：Youden 最优阈值仅 0.18，在 0.5 阈值下 Sensitivity 仅 0.39
3. EfficientNetV2-S 特异性最强（0.931），但 Sensitivity 最低（0.662），保守型模型
4. ConvNeXt-Tiny V2 内外不一致：BUSBRA 排名第 2，但 BUSI 排名第 5，泛化能力存疑
5. Swin-Tiny 内外一致偏弱：两个数据集都排名靠后

## 六、新主线集成建议

| 方案 | 模型组合 | 预期优势 |
|---|---|---|
| 方案 A（推荐） | ConvNeXt-Small + ConvNeXt-Tiny V1 + EfficientNetV2-S | 外部泛化最强 + 内部最高 + 特异性互补 |
| 方案 B | ConvNeXt-Small + ConvNeXt-Tiny V1 + ConvNeXt-Tiny V2 | 三个 ConvNeXt 变体，稳定性好 |
| 方案 C | 全部 6 模型 | 最大异构性，但需重新搜索权重 |

建议排除：DenseNet-121（阈值偏移严重）和 Swin-Tiny（内外一致偏弱）

## 七、Checkpoint 文件路径

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

## 八、推理配置文件路径

- configs/inference/convnext_tiny_timm_recipe_5fold_tta_identity.yml
- configs/inference/convnextv2_tiny_timm_recipe_5fold_tta_identity.yml
- configs/inference/convnext_small_timm_recipe_5fold_tta_crop_sweep.yml
- configs/inference/efficientnetv2_s_5fold_identity.yml
- configs/inference/densenet121_5fold_tta_identity.yml
- configs/inference/swin_tiny_timm_recipe_5fold_tta_identity.yml

## 九、原始评估报告路径

- artifacts/reports/busi_eval_convnext_v1.json
- artifacts/reports/busi_eval_convnextv2_v2.json
- artifacts/reports/busi_eval_convnext_small_identity.json
- artifacts/reports/busi_eval_efficientnetv2_s_identity.json
- artifacts/reports/busi_eval_densenet121_identity.json
- artifacts/reports/busi_eval_swin_tiny_identity.json
