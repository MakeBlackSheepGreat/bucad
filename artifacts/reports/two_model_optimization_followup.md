# 两模型集成后续优化报告

日期：2026-04-25

## 实验边界

- 本轮只优化当前两模型方向：`ConvNeXt-Tiny` + `EfficientNetV2-S`。
- 没有使用 BUSI 参与训练；BUSI 只用于外部评估、阈值分析和候选运行点比较。
- 本报告新增的是候选配置，不直接替换 `configs/inference/demo.yml`。是否切换 Demo 主线，需要根据 AUC、Sensitivity 和 Specificity 的取舍决定。

## 当前主线基准

当前 Demo 使用的是 ConvNeXt-Tiny 五折 crop-sweep TTA + EfficientNetV2-S 五折 identity TTA：

| 方案 | 配置 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | 混淆矩阵 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 当前两模型正式方案 | `configs/inference/ensemble_effnet_convnext_optimized.yml` | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | TN 387 / FP 50 / FN 42 / TP 168 |

这个基准已经处在比较稳定的位置，后续优化不能只看单一 AUC，还要看是否牺牲敏感性。

## 补齐的单模型外部评估

| 单模型方案 | 报告 | AUC | 默认阈值 Sensitivity | 默认阈值 Specificity | 默认阈值 Accuracy | Youden 推荐阈值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EfficientNetV2-S 五折 identity | `artifacts/reports/busi_efficientnetv2_s_5fold_identity.json` | 0.8982 | 0.6619 | 0.9314 | 0.8439 | 0.39 |
| EfficientNetV2-S 五折 hflip | `artifacts/reports/busi_efficientnetv2_s_5fold_hflip.json` | 0.8997 | 0.6810 | 0.9291 | 0.8485 | 0.36 |

EfficientNetV2-S 单独看时 hflip 略好，但在两模型集成里，identity 与 ConvNeXt 的互补性更好。

## 权重与 TTA 搜索结论

本轮离线搜索了以下组合：

- EfficientNetV2-S：identity、hflip。
- ConvNeXt-Tiny：identity、hflip、rotate5、crop90 hflip、crop100 hflip、crop-sweep。
- 融合方式：普通概率加权、logit 加权。
- 权重步长：0.001。

搜索结果显示：logit 融合没有带来 AUC 优势；可直接落地的普通概率加权仍然最稳。

## 实跑候选结果

| 方案 | 配置 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | 混淆矩阵 | 主要变化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 当前两模型正式方案 | `configs/inference/ensemble_effnet_convnext_optimized.yml` | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | TN 387 / FP 50 / FN 42 / TP 168 | 当前 Demo 基准 |
| 保守微调候选 | `configs/inference/ensemble_effnet_convnext_crop_sweep_weight0569.yml` | 0.9152 | 0.399 | 0.8000 | 0.8856 | 0.8578 | TN 387 / FP 50 / FN 42 / TP 168 | 只把 ConvNeXt 权重从 0.573 微调到 0.569，收益极小 |
| AUC/特异性候选 | `configs/inference/ensemble_effnet_convnext_hflip_weight0511.yml` | 0.9159 | 0.385 | 0.7905 | 0.8947 | 0.8609 | TN 391 / FP 46 / FN 44 / TP 166 | AUC、Specificity、Accuracy 提升，但 Sensitivity 下降 |

## 阈值取舍

`configs/inference/ensemble_effnet_convnext_hflip_weight0511.yml` 的关键阈值点如下：

| 阈值 | Sensitivity | Specificity | Accuracy | Youden J | 混淆矩阵 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.385 | 0.7905 | 0.8947 | 0.8609 | 0.6852 | TN 391 / FP 46 / FN 44 / TP 166 |
| 0.395 | 0.7857 | 0.8993 | 0.8624 | 0.6850 | TN 393 / FP 44 / FN 45 / TP 165 |
| 0.400 | 0.7810 | 0.9016 | 0.8624 | 0.6826 | TN 394 / FP 43 / FN 46 / TP 164 |

如果展示目标是“更少误报、更高准确率”，AUC/特异性候选更好；如果展示目标是“尽量不漏诊”，当前 crop-sweep 主线更合适。

## 结论

- 当前两模型方案已经接近局部最优，单靠权重和 TTA 微调，AUC 提升空间约为 `0.0008`，不是质变。
- `ensemble_effnet_convnext_hflip_weight0511.yml` 是本轮最好的 AUC 候选：AUC 从 `0.9151` 提升到 `0.9159`，Specificity 从 `0.8856` 提升到 `0.8947`，Accuracy 从 `0.8578` 提升到 `0.8609`。
- 代价是 Sensitivity 从 `0.8000` 降到 `0.7905`。乳腺肿瘤筛查场景通常更重视漏诊风险，因此不建议立刻替换当前 Demo 主线。
- `ensemble_effnet_convnext_crop_sweep_weight0569.yml` 可视为安全微调版，但指标几乎不变，不值得单独作为重大优化宣传。

## 下一步建议

1. Demo 主线暂时保持当前 crop-sweep 方案，除非明确要展示更高 AUC/Specificity。
2. 如果要继续提高 AUC，下一步优先做 `seed ensemble` 或基于 BUSBRA OOF 的校准，而不是继续在 BUSI 上反复搜阈值。
3. 如果要提高 Sensitivity，应继续围绕当前 crop-sweep 方案做阈值和高召回模式，而不是切到 hflip 权重候选。
4. 后续报告中可以把 hflip 候选作为“特异性优先运行点”，但不要把它包装成全面优于当前主线。
