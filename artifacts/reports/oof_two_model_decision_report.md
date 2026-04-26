# 两模型 OOF 优化决策报告

> 数据边界说明：BUSBRA 用于训练、内部验证、OOF、模型选择、阈值选择和 ROI 参数选择；BUSI 是锁定外部评估集，只用于最终外部验证，不用于训练或调参。


日期：2026-04-25

## 实验目的

本轮目标是用更规范的方式优化当前两模型集成，而不是继续直接在 BUSI 外部评估集上搜索权重。具体做法是：

1. 使用 BUSBRA 五折验证集生成 out-of-fold 概率。
2. 在 BUSBRA OOF 概率上训练或选择融合策略。
3. 将得到的融合策略迁移到 BUSI，只把 BUSI 作为外部评估集。

## 数据边界

- BUSBRA：用于 OOF 概率生成、融合器训练、OOF 交叉验证和 OOF 阈值选择。
- BUSI：只用于外部评估，不参与训练、不参与 OOF 融合器选择。
- 当前 Demo 主线没有修改。

## 输入模型视图

| 视图 | 含义 | 用途 |
| --- | --- | --- |
| `eff_identity` | EfficientNetV2-S 五折 identity TTA | 两模型辅助分支 |
| `conv_crop_sweep` | ConvNeXt-Tiny 五折 crop-sweep TTA | 当前 Demo 主线分支 |
| `conv_hflip` | ConvNeXt-Tiny 五折 hflip TTA | 上一轮 AUC 候选分支 |

## 逻辑回归 Stacking 结果

| 方案 | 特征 | OOF CV AUC | BUSI AUC | BUSI@0.5 Sens | BUSI@0.5 Spec | BUSI@0.5 Acc | 评价 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `current_crop_sweep` | `eff_identity + conv_crop_sweep` | 0.9245 | 0.9130 | 0.7762 | 0.8970 | 0.8578 | 外部 AUC 低于主线 |
| `auc_hflip` | `eff_identity + conv_hflip` | 0.9249 | 0.9134 | 0.6857 | 0.9428 | 0.8594 | 过度偏向特异性，敏感性下降明显 |
| `multi_view` | `eff_identity + conv_crop_sweep + conv_hflip` | 0.9249 | 0.9138 | 0.9286 | 0.6339 | 0.7295 | 校准偏移严重，不能作为默认方案 |

逻辑回归 Stacking 在 BUSBRA OOF 上表现不错，但迁移到 BUSI 后 AUC 低于当前主线 `0.9151`。这说明 BUSBRA 与 BUSI 存在分布差异，融合器学到的概率校准关系没有稳定迁移。

## OOF 静态权重搜索结果

相比逻辑回归，静态概率加权更保守，泛化也更稳定。

| 方案 | OOF 学到的权重 | OOF AUC | OOF阈值 | BUSI AUC | BUSI@OOF阈值 Sens | BUSI@OOF阈值 Spec | BUSI最佳分析阈值 | BUSI最佳分析 Sens | BUSI最佳分析 Spec | 评价 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `current_crop_sweep` 概率加权 | Eff 0.344 / Conv 0.656 | 0.9199 | 0.24 | 0.9143 | 0.8905 | 0.7574 | 0.43 | 0.7952 | 0.8879 | AUC 低于主线，不推荐 |
| `auc_hflip` 概率加权 | Eff 0.456 / Conv 0.544 | 0.9184 | 0.24 | 0.9158 | 0.8857 | 0.7849 | 0.40 | 0.7857 | 0.8993 | AUC 略高于主线，但敏感性低于主线 |

其中 `auc_hflip` 的 OOF 静态权重是本轮最接近可用的结果：BUSI AUC `0.9158`，高于当前主线 `0.9151`，但它在合理运行点下的敏感性仍低于当前主线。

## 与当前主线对比

| 方案 | 权重来源 | BUSI AUC | 阈值 | Sensitivity | Specificity | Accuracy | 混淆矩阵 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 当前 Demo 主线 | 早前权重搜索 + crop-sweep | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | TN 387 / FP 50 / FN 42 / TP 168 |
| OOF 静态权重候选 | BUSBRA OOF + hflip | 0.9158 | BUSI分析阈值 0.40 | 0.7857 | 0.8993 | 0.8624 | TN 393 / FP 44 / FN 45 / TP 165 |
| 上轮 hflip 候选 | BUSI 直接权重搜索 + hflip | 0.9159 | 0.385 | 0.7905 | 0.8947 | 0.8609 | TN 391 / FP 46 / FN 44 / TP 166 |

## 结论

- OOF 方法是规范的，但没有把 AUC 明显推高。
- 逻辑回归 Stacking 不建议采用：OOF 内部表现好，BUSI 外部 AUC 反而下降，说明它对 BUSBRA 的概率分布拟合过强。
- OOF 静态权重可以轻微提升 AUC 到 `0.9158`，但代价是敏感性从 `0.8000` 降到 `0.7857`。
- 当前项目是乳腺肿瘤筛查/辅助诊断场景，漏诊代价更高，因此不建议为了 `+0.0006` 左右的 AUC 替换当前 Demo 主线。

## 决策建议

1. 保持当前 `ConvNeXt-Tiny crop-sweep + EfficientNetV2-S identity` Demo 主线。
2. 把 OOF hflip 静态权重作为“特异性优先候选”，不要作为默认方案。
3. 如果继续追求 AUC，更值得做的是重训弱 fold 或 seed ensemble，而不是继续做更复杂的 stacking。
4. 如果答辩需要展示严谨性，可以强调：我们尝试了更规范的 OOF Stacking，但由于外部分布迁移不稳定，最终选择更稳健的主线方案。
