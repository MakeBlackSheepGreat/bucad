# Swin-Tiny 五折训练与外部评估

日期：2026-04-25

## 实验目标

本轮实验补齐 `swin_tiny_patch4_window7_224` 的五折训练，并评估它作为独立模型和后续集成候选的价值。训练仍严格使用 BUSBRA 训练/内部验证划分，BUSI 只用于外部评估和运行点分析，不参与训练。

## 训练配置

| 项目 | 设置 |
| --- | --- |
| Backbone | `swin_tiny_patch4_window7_224` |
| 预训练 | timm ImageNet 预训练 |
| 输入 | 224 |
| 插值与裁剪 | bicubic，`crop_pct=0.90` |
| 归一化 | mean `[0.485, 0.456, 0.406]`，std `[0.229, 0.224, 0.225]` |
| 超声预处理 | CLAHE |
| 增强 | 水平翻转 |
| 优化器 | AdamW |
| 学习率 | `5e-5` |
| 权重衰减 | `0.05` |
| 类别权重 | balanced |
| 调度 | warmup + cosine |
| checkpoint 选择 | validation AUC 最优 |
| 折数 | 5 |

## BUSBRA 五折内部验证

| Fold | Best Epoch | AUC | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 30 | 0.9053 | 0.7541 | 0.8933 | 0.8480 | 0.7731 | 0.7635 | TN 226 / FP 27 / FN 30 / TP 92 |
| 2 | 29 | 0.9291 | 0.7851 | 0.9134 | 0.8720 | 0.8120 | 0.7983 | TN 232 / FP 22 / FN 26 / TP 95 |
| 3 | 11 | 0.9300 | 0.7934 | 0.8622 | 0.8400 | 0.7328 | 0.7619 | TN 219 / FP 35 / FN 25 / TP 96 |
| 4 | 16 | 0.9053 | 0.7377 | 0.8854 | 0.8373 | 0.7563 | 0.7469 | TN 224 / FP 29 / FN 32 / TP 90 |
| 5 | 14 | 0.8998 | 0.7273 | 0.8819 | 0.8320 | 0.7458 | 0.7364 | TN 224 / FP 30 / FN 33 / TP 88 |
| Mean | - | 0.9139 | 0.7595 | 0.8872 | 0.8459 | - | - | - |

内部验证看起来稳定，AUC 均值达到 0.9139；但 Sensitivity 均值只有 0.7595，说明 Swin-Tiny 在 BUSBRA 内部验证上仍偏保守，对恶性样本召回不算突出。

## BUSI 外部评估

| 配置 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Swin-Tiny fold1 | 0.8721 | 0.50 | 0.7048 | 0.8902 | 0.8300 | 0.7551 | 0.7291 | TN 389 / FP 48 / FN 62 / TP 148 |
| Swin-Tiny 5-fold + hflip TTA | 0.8867 | 0.50 | 0.7476 | 0.8650 | 0.8269 | 0.7269 | 0.7371 | TN 378 / FP 59 / FN 53 / TP 157 |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.8971 | 0.50 | 0.7762 | 0.8833 | 0.8485 | 0.7617 | 0.7689 | TN 386 / FP 51 / FN 47 / TP 163 |

## BUSI 最优 Youden 运行点

| 配置 | AUC | 最优阈值 | Sensitivity | Specificity | Accuracy | Youden J | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Swin-Tiny fold1 | 0.8721 | 0.18 | 0.7524 | 0.8696 | 0.8315 | 0.6219 | 0.7349 | 0.7435 | TN 380 / FP 57 / FN 52 / TP 158 |
| Swin-Tiny 5-fold + hflip TTA | 0.8867 | 0.67 | 0.7190 | 0.9199 | 0.8547 | 0.6390 | 0.8118 | 0.7626 | TN 402 / FP 35 / FN 59 / TP 151 |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.8971 | 0.50 | 0.7762 | 0.8833 | 0.8485 | 0.6595 | 0.7617 | 0.7689 | TN 386 / FP 51 / FN 47 / TP 163 |

## 与当前候选模型对比

| 模型/系统 | AUC | 推荐阈值 | Sensitivity | Precision | F1-Score | Specificity | Accuracy | Youden J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.8971 | 0.50 | 0.7762 | - | - | 0.8833 | 0.8485 | 0.6595 |
| ConvNeXt-Tiny 5-fold + crop_sweep TTA | 0.9054 | 0.46 | 0.8048 | - | - | 0.8719 | 0.8501 | 0.6766 |
| EfficientNetV2-S + DenseNet121 + ConvNeXt-Tiny 三模型集成 | 0.9138 | 0.32 | 0.8095 | - | - | 0.8627 | 0.8454 | 0.6722 |

## 结论

- Swin-Tiny 五折相对 fold1 有明确提升：AUC 从 0.8721 提升到 0.8867，加入多裁剪 TTA 后进一步到 0.8971。
- 但 Swin-Tiny 仍低于 ConvNeXt-Tiny：AUC 差距约 0.0083，Youden J 差距约 0.0171。
- Swin-Tiny 也明显低于当前三模型集成：AUC 差距约 0.0167。
- 当前不建议把 Swin-Tiny 作为主模型替代 ConvNeXt-Tiny，也不建议优先投入大量时间继续微调 Swin-Tiny。
- 如果后续要用 Swin-Tiny，比较合理的定位是“低权重补充成员”，只在权重搜索证明它能提升三模型集成后再加入。

## 产物

- 训练配置：`configs/classifier/swin_tiny_patch4_window7_224_timm_recipe.yml`
- 五折推理配置：`configs/inference/swin_tiny_timm_recipe_5fold.yml`
- 多裁剪 TTA 配置：`configs/inference/swin_tiny_timm_recipe_5fold_tta_crop_sweep.yml`
- BUSI 五折评估：`artifacts/reports/busi_swin_tiny_timm_recipe_5fold.json`
- BUSI 多裁剪评估：`artifacts/reports/busi_swin_tiny_timm_recipe_5fold_tta_crop_sweep.json`
