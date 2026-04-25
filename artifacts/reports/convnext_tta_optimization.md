# ConvNeXt-Tiny TTA 优化实验

日期：2026-04-25

## 目标

本轮实验只优化 ConvNeXt-Tiny 五折模型的测试时增强（TTA）。训练边界保持不变：BUSBRA 只用于训练和内部验证，BUSI 只用于外部评估与运行点分析，不参与训练。

## 实现调整

- 推理服务新增 `classifier_tta_variants`，可以显式配置 identity、水平翻转、轻微旋转和不同 `crop_pct` 的 TTA 组合。
- 推理报告新增 `runtime_summary`，记录输入尺寸、裁剪比例和 TTA 变体，方便复现实验。
- BUSI 评估在传入 `--output` 时会额外生成同名阈值分析报告，避免多组实验只覆盖一个 `threshold_analysis.md`。
- ConvNeXt-Tiny TTA 推理改为按 TTA batch 一次前向，并读取 `device: auto` 使用 CUDA；本机评估使用 NVIDIA GPU 完成。

## 候选 TTA 配置

| 配置 | TTA 变体 | 单轮耗时 |
| --- | --- | ---: |
| identity | `crop_pct=0.95` 原图 | 39.5s |
| hflip | `crop_pct=0.95` 原图 + 水平翻转 | 46.0s |
| crop90_hflip | `crop_pct=0.90` 原图 + 水平翻转 | 45.1s |
| crop100_hflip | `crop_pct=1.00` 原图 + 水平翻转 | 45.2s |
| crop_sweep | `crop_pct=0.90/0.95/1.00`，每个裁剪配原图 + 水平翻转 | 73.7s |
| rotate5 | `crop_pct=0.95` 原图 + 水平翻转 + `±5°` 旋转 | 57.9s |

## BUSI 外部评估：默认阈值 0.50

| 配置 | AUC | Sensitivity | Specificity | Accuracy | Confusion |
| --- | ---: | ---: | ---: | ---: | --- |
| identity | 0.8991 | 0.7381 | 0.8810 | 0.8346 | TN 385 / FP 52 / FN 55 / TP 155 |
| hflip | 0.9041 | 0.7381 | 0.8924 | 0.8423 | TN 390 / FP 47 / FN 55 / TP 155 |
| crop90_hflip | 0.9018 | 0.7524 | 0.8810 | 0.8393 | TN 385 / FP 52 / FN 52 / TP 158 |
| crop100_hflip | 0.8994 | 0.7762 | 0.8581 | 0.8315 | TN 375 / FP 62 / FN 47 / TP 163 |
| crop_sweep | **0.9054** | **0.7714** | 0.8879 | **0.8501** | TN 388 / FP 49 / FN 48 / TP 162 |
| rotate5 | 0.9048 | 0.7381 | **0.8970** | 0.8454 | TN 392 / FP 45 / FN 55 / TP 155 |

## BUSI 外部评估：Youden 最优运行点

| 配置 | 阈值 | AUC | Sensitivity | Specificity | Accuracy | Youden J | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| identity | 0.54 | 0.8991 | 0.7381 | 0.9085 | 0.8532 | 0.6466 | TN 397 / FP 40 / FN 55 / TP 155 |
| hflip | 0.47 | 0.9041 | 0.7667 | 0.8879 | 0.8485 | 0.6545 | TN 388 / FP 49 / FN 49 / TP 161 |
| crop90_hflip | 0.26 | 0.9018 | **0.8619** | 0.7941 | 0.8161 | 0.6560 | TN 347 / FP 90 / FN 29 / TP 181 |
| crop100_hflip | 0.82 | 0.8994 | 0.6857 | **0.9565** | **0.8686** | 0.6422 | TN 418 / FP 19 / FN 66 / TP 144 |
| crop_sweep | 0.46 | **0.9054** | 0.8048 | 0.8719 | 0.8501 | **0.6766** | TN 381 / FP 56 / FN 41 / TP 169 |
| rotate5 | 0.40 | 0.9048 | 0.7810 | 0.8696 | 0.8408 | 0.6505 | TN 380 / FP 57 / FN 46 / TP 164 |

## 与现有集成基线对比

| 模型/系统 | AUC | 推荐阈值 | Sensitivity | Specificity | Accuracy | Youden J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNetV2-S 五折 + TTA | 0.8997 | 0.33 | 0.8000 | 0.8764 | **0.8516** | 0.6764 |
| EfficientNetV2-S + DenseNet121 混合集成 | 0.9052 | 0.27 | **0.8095** | 0.8719 | **0.8516** | **0.6814** |
| ConvNeXt-Tiny 五折 + crop_sweep TTA | **0.9054** | 0.46 | 0.8048 | 0.8719 | 0.8501 | 0.6766 |

## 结论

- ConvNeXt-Tiny 对 TTA 很敏感。无 TTA 时 AUC 为 0.8991，加入水平翻转后提升到 0.9041。
- 单一裁剪比例并不稳定：`crop_pct=0.90` 和 `crop_pct=1.00` 单独使用都低于标准 `0.95 + hflip`。
- 多裁剪平均最有效。`crop_sweep` 将 AUC 提升到 0.9054，略高于当前 EfficientNetV2-S + DenseNet121 混合集成的 0.9052。
- 但从运行点看，当前混合集成仍有更高 Youden J、更高 Sensitivity 和更高 Accuracy。因此 ConvNeXt-Tiny 暂时不应直接替换演示默认模型。
- 推荐把 `configs/inference/convnext_tiny_timm_recipe_5fold_tta_crop_sweep.yml` 作为 ConvNeXt-Tiny 后续主模型候选配置，再进入 Swin-Tiny 低权重集成搜索。

## 后续建议

1. 先固定 ConvNeXt-Tiny 的候选 TTA 为 `crop_sweep`，避免继续在单模型 TTA 上做低收益搜索。
2. 下一步训练或补齐 Swin-Tiny 五折后，测试 ConvNeXt-Tiny crop_sweep 与 Swin-Tiny 的低权重集成。
3. 若最终目标偏筛查场景，可以优先比较 Youden 运行点附近的 Sensitivity，而不是只看默认 0.50 阈值。
