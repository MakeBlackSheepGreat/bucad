# 小规模弱折重训外部验证报告

> 数据边界说明：BUSBRA 用于训练、内部验证、OOF、模型选择、阈值选择和 ROI 参数选择；BUSI 是锁定外部评估集，只用于最终外部验证，不用于训练或调参。


日期：2026-04-25

## 实验目的

本轮实验针对当前两模型主线中的弱折做小规模重训，目标是用较低成本提高整体 AUC。

主线保持不变：

- 主模型：`ConvNeXt-Tiny`
- 辅助模型：`EfficientNetV2-S`
- 当前 Demo 配置：`configs/inference/demo.yml`
- 当前正式推理配置：`configs/inference/ensemble_effnet_convnext_optimized.yml`

本轮没有修改 `demo.yml`，也没有替换主线权重。

## 数据边界

- BUSBRA：用于训练和内部五折验证。
- BUSI：只用于外部评估，不参与训练、不参与权重选择。
- 所有重训 checkpoint 都以新文件名保存，不覆盖已有主线 checkpoint。

## 弱折定位

| 模型 | Fold | 原内部 AUC | 原 Sensitivity | 原 Specificity | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny | 4 | 0.8875 | 0.7623 | 0.8577 | ConvNeXt 五折中最低 |
| EfficientNetV2-S | 5 | 0.8523 | 0.6116 | 0.9016 | EfficientNet 五折中最低 |

## 重训配置

| 配置 | 目标 | 输出 checkpoint |
| --- | --- | --- |
| `configs/classifier/convnext_tiny_timm_recipe_seed123.yml` | 使用相同训练配方，只更换 seed，重训 ConvNeXt-Tiny fold4 | `artifacts/checkpoints/convnext_tiny_timm_recipe_seed123_fold4.pt` |
| `configs/classifier/efficientnetv2_s_bestauc_seed123.yml` | 使用 seed123 和 best_auc 策略，重训 EfficientNetV2-S fold5 | `artifacts/checkpoints/efficientnetv2_s_bestauc_seed123_fold5.pt` |

## 内部验证结果

| 模型 | Fold | 原内部 AUC | 重训内部 AUC | 变化 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny | 4 | 0.8875 | 0.9118 | +0.0243 | 内部验证明显改善 |
| EfficientNetV2-S | 5 | 0.8523 | 0.9007 | +0.0484 | 内部验证明显改善 |

如果只看 BUSBRA 内部验证，本轮重训似乎有效。但这不是最终判断标准，必须看 BUSI 外部泛化。

## BUSI 单折外部验证

| 模型 | Fold | 原 BUSI AUC | 重训 BUSI AUC | 变化 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny crop-sweep | 4 | 0.8901 | 0.8845 | -0.0056 | 内部改善没有迁移到外部 |
| EfficientNetV2-S identity | 5 | 0.8633 | 0.8527 | -0.0106 | 内部改善没有迁移到外部 |

单折外部评估已经说明：弱折重训更贴合 BUSBRA 对应验证折，但对 BUSI 外部域排序能力下降。

## BUSI 集成外部验证

| 方案 | 配置 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | 混淆矩阵 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 当前主线 | `configs/inference/ensemble_effnet_convnext_optimized.yml` | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |
| 只替换 ConvNeXt fold4 | `configs/inference/ensemble_effnet_convnext_retrain_conv4.yml` | 0.8712 | 0.399 | 0.7571 | 0.8398 | 0.8130 | 0.6943 | 0.7244 | TN 367 / FP 70 / FN 51 / TP 159 |
| 只替换 EfficientNet fold5 | `configs/inference/ensemble_effnet_convnext_retrain_eff5.yml` | 0.8708 | 0.399 | 0.7762 | 0.8558 | 0.8300 | 0.7212 | 0.7477 | TN 374 / FP 63 / FN 47 / TP 163 |
| 同时替换两个弱折 | `configs/inference/ensemble_effnet_convnext_retrain_conv4_eff5.yml` | 0.8701 | 0.399 | 0.7524 | 0.8581 | 0.8238 | 0.7182 | 0.7349 | TN 375 / FP 62 / FN 52 / TP 158 |
| hflip 候选同时替换两个弱折 | `configs/inference/ensemble_effnet_convnext_hflip_weight0511_retrain_conv4_eff5.yml` | 0.9140 | 0.385 | 0.7571 | 0.8970 | 0.8516 | 0.7794 | 0.7681 | TN 392 / FP 45 / FN 51 / TP 159 |

集成结果进一步确认：重训弱折不适合并入当前主线。

## 过拟合判断

本轮现象符合典型外部泛化过拟合：

1. BUSBRA 内部验证 AUC 明显提高。
2. BUSI 单折外部 AUC 下降。
3. 替换进集成后，整体外部 AUC 明显下降。
4. 重训权重更像是拟合了 BUSBRA fold 的局部分布，而不是学到更稳健的跨数据集特征。

因此不能用“内部弱折修好”作为替换主线的依据。

## 决策

- 保留当前主线权重。
- 不把 `convnext_tiny_timm_recipe_seed123_fold4.pt` 并入 Demo。
- 不把 `efficientnetv2_s_bestauc_seed123_fold5.pt` 并入 Demo。
- 本轮重训配置和报告保留为负结果证据，后续用于解释为什么没有盲目追求内部验证指标。

## 后续优化方向

下一步更建议转向 ROI 裁剪/病灶区域引导，并结合 OOF 机制降低过拟合风险：

1. **ROI 裁剪/病灶区域引导**：利用分割 mask、弱定位或 Grad-CAM 生成肿块区域裁剪图，让模型减少背景噪声依赖。
2. **原图 + ROI 双视图融合**：保留完整超声图上下文，同时增加病灶局部细节输入。
3. **OOF 机制约束选择**：所有融合权重、阈值和二级融合器优先在 BUSBRA OOF 上确定，BUSI 只做外部验证。
4. **外部泛化优先**：后续不再只根据 BUSBRA fold 指标替换权重，必须同时检查 BUSI 单模型和集成外部表现。

## 总结

本轮小规模重训给出了一个重要结论：当前项目的主要瓶颈不是某个弱 fold 训练不够，而是 BUSBRA 到 BUSI 的外部分布迁移。继续简单重训弱折的性价比不高，后续应从输入信息质量和泛化约束入手。
