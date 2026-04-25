# ConvNeXt-Small 主模型升级旁路实验

日期：2026-04-25

## 实验边界

- 本实验只测试升级主模型的可行性，不修改 `configs/inference/demo.yml`、README 或当前主线集成方案。
- 训练数据仍只使用 BUSBRA fold1 训练划分；BUSI 只用于外部评估和阈值分析。
- 对比对象采用 fold1 单模型，避免拿单折模型和已经优化好的五折/集成模型直接比较。

## 新增配置

- 训练配置：`configs/classifier/convnext_small_timm_recipe.yml`
- 标准 TTA 评估配置：`configs/inference/convnext_small_timm_recipe_fold1.yml`
- crop-sweep TTA 评估配置：`configs/inference/convnext_small_timm_recipe_fold1_tta_crop_sweep.yml`
- 公平对照配置：`configs/inference/convnext_tiny_timm_recipe_fold1_tta_crop_sweep.yml`

## 训练结果

| 模型 | fold | 内部验证 AUC | Sensitivity | Specificity | Accuracy | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny timm recipe | 1 | 0.9259 | 0.7213 | 0.9328 | 0.8640 | 已有基准 |
| ConvNeXt-Small timm recipe | 1 | 0.9118 | 0.7295 | 0.9091 | 0.8507 | 本轮新增 |

训练观察：ConvNeXt-Small 在后半程训练损失接近 0，说明参数量增大后更容易记忆训练集，存在过拟合风险。

## BUSI 外部评估

| 模型 | TTA | AUC | 阈值 | Sensitivity | Specificity | Accuracy | 混淆矩阵 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny fold1 | hflip | 0.8943 | 0.50 | 0.7762 | 0.8741 | 0.8423 | TN 382 / FP 55 / FN 47 / TP 163 |
| ConvNeXt-Tiny fold1 | crop-sweep | 0.8953 | 0.50 | 0.7952 | 0.8696 | 0.8454 | TN 380 / FP 57 / FN 43 / TP 167 |
| ConvNeXt-Small fold1 | hflip | 0.8947 | 0.50 | 0.7667 | 0.8581 | 0.8284 | TN 375 / FP 62 / FN 49 / TP 161 |
| ConvNeXt-Small fold1 | crop-sweep | 0.8991 | 0.50 | 0.7857 | 0.8673 | 0.8408 | TN 379 / FP 58 / FN 45 / TP 165 |

## Youden 运行点

| 模型 | TTA | AUC | 推荐阈值 | Sensitivity | Specificity | Accuracy | Youden J |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny fold1 | crop-sweep | 0.8953 | 0.49 | 0.7952 | 0.8719 | 0.8470 | 0.6671 |
| ConvNeXt-Small fold1 | crop-sweep | 0.8991 | 0.28 | 0.8381 | 0.8284 | 0.8315 | 0.6665 |

## 结论

- ConvNeXt-Small 不是失败方向：在同为 fold1 + crop-sweep TTA 的公平对照下，BUSI AUC 从 `0.8953` 提升到 `0.8991`，提升约 `0.0037`。
- 但它还不能替代当前主线：当前正式 demo 是 ConvNeXt-Tiny + EfficientNetV2-S 两模型五折优化集成，AUC `0.9151`，明显高于单折 Small。
- Small 的内部验证 AUC 低于 Tiny，且训练损失后期接近 0，说明更大模型需要更强正则或更保守训练策略。
- 建议暂不改主线；如果继续推进，下一步应训练 ConvNeXt-Small 五折，并配合更强正则、较低学习率、早停或 Mixup/CutMix，再和当前五折 Tiny 主模型公平比较。
