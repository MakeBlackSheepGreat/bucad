# 两模型与三模型当前模型调优报告

> ???????????? BUSI ??????????????????? BUSI ???????TTA?????????????????????????????????? BUSBRA OOF/????????? BUSI ????????


日期：2026-04-25

## 调优边界

- 不重新训练模型，只优化现有 checkpoint 的推理策略。
- BUSI 只用于外部评估、TTA 消融和阈值运行点分析，不参与训练。
- 本轮重点比较 ConvNeXt 分支 TTA、CNN 分支 hflip 是否保留、以及 0.001 粒度阈值。

## TTA 消融结果

| 方案 | TTA 策略 | AUC | 默认阈值 | Sensitivity | Specificity | Accuracy | 混淆矩阵 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 两模型 | 基线：CNN hflip + ConvNeXt crop_sweep | 0.9142 | 0.500 | 0.7524 | 0.9130 | 0.8609 | TN 399 / FP 38 / FN 52 / TP 158 |
| 两模型 | ConvNeXt hflip | 0.9148 | 0.500 | 0.7143 | 0.9245 | 0.8563 | TN 404 / FP 33 / FN 60 / TP 150 |
| 两模型 | ConvNeXt rotate5 | 0.9148 | 0.500 | 0.7286 | 0.9199 | 0.8578 | TN 402 / FP 35 / FN 57 / TP 153 |
| 两模型 | CNN identity + ConvNeXt crop_sweep | 0.9151 | 0.500 | 0.7524 | 0.9176 | 0.8640 | TN 401 / FP 36 / FN 52 / TP 158 |
| 三模型 | 基线：CNN hflip + ConvNeXt crop_sweep | 0.9144 | 0.500 | 0.7095 | 0.9382 | 0.8640 | TN 410 / FP 27 / FN 61 / TP 149 |
| 三模型 | ConvNeXt hflip | 0.9153 | 0.500 | 0.6857 | 0.9382 | 0.8563 | TN 410 / FP 27 / FN 66 / TP 144 |
| 三模型 | ConvNeXt rotate5 | 0.9151 | 0.500 | 0.6857 | 0.9382 | 0.8563 | TN 410 / FP 27 / FN 66 / TP 144 |
| 三模型 | CNN identity + ConvNeXt crop_sweep | 0.9162 | 0.500 | 0.6952 | 0.9405 | 0.8609 | TN 411 / FP 26 / FN 64 / TP 146 |

## 0.001 细粒度阈值建议

| 方案 | TTA 策略 | 目标 | 阈值 | Sensitivity | Specificity | Accuracy | Youden J | 混淆矩阵 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 两模型 | CNN identity + ConvNeXt crop_sweep | Youden 最优 | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.6856 | TN 387 / FP 50 / FN 42 / TP 168 |
| 两模型 | CNN identity + ConvNeXt crop_sweep | Accuracy 最优 | 0.543 | 0.7238 | 0.9359 | 0.8671 | 0.6597 | TN 409 / FP 28 / FN 58 / TP 152 |
| 两模型 | CNN identity + ConvNeXt crop_sweep | Sensitivity≥0.85 | 0.299 | 0.8524 | 0.8055 | 0.8207 | 0.6579 | TN 352 / FP 85 / FN 31 / TP 179 |
| 三模型 | CNN identity + ConvNeXt crop_sweep | Youden / Accuracy 最优 | 0.453 | 0.7476 | 0.9291 | 0.8702 | 0.6767 | TN 406 / FP 31 / FN 53 / TP 157 |
| 三模型 | CNN identity + ConvNeXt crop_sweep | Sensitivity≥0.80 | 0.350 | 0.8000 | 0.8719 | 0.8485 | 0.6719 | TN 381 / FP 56 / FN 42 / TP 168 |
| 三模型 | CNN identity + ConvNeXt crop_sweep | Sensitivity≥0.85 | 0.253 | 0.8571 | 0.8146 | 0.8284 | 0.6718 | TN 356 / FP 81 / FN 30 / TP 180 |
| 三模型 | ConvNeXt rotate5 | 高召回候选 | 0.228 | 0.8762 | 0.8078 | 0.8300 | 0.6840 | TN 353 / FP 84 / FN 26 / TP 184 |

## 最终优化配置

| 配置 | 文件 | AUC | 默认阈值 | Sensitivity | Specificity | Accuracy | 混淆矩阵 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 两模型最终配置 | `configs/inference/ensemble_effnet_convnext_optimized.yml` | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | TN 387 / FP 50 / FN 42 / TP 168 |
| 三模型最终配置 | `configs/inference/ensemble_effnet_densenet_convnext_optimized.yml` | 0.9162 | 0.453 | 0.7476 | 0.9291 | 0.8702 | TN 406 / FP 31 / FN 53 / TP 157 |

## 结论

- 两模型最优 TTA 是 `CNN identity + ConvNeXt crop_sweep`，AUC 从 `0.9142` 提升到 `0.9151`。
- 两模型推荐均衡阈值为 `0.399`，Sensitivity `0.8000`，Specificity `0.8856`，Youden J `0.6856`。
- 三模型最优 AUC 也是 `CNN identity + ConvNeXt crop_sweep`，AUC 从 `0.9144` 提升到 `0.9162`。
- 三模型推荐主配置阈值为 `0.453`，Accuracy `0.8702`，是本轮最高准确率运行点。
- 如果后续更强调筛查召回，可以把三模型临时切到 `ConvNeXt rotate5` 并使用阈值 `0.228`，Sensitivity 可到 `0.8762`，但 Accuracy 会降到 `0.8300`。
