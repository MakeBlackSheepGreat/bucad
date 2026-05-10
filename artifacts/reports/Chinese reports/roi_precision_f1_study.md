# ROI Precision / F1 优化研究

日期：2026-04-26

## 目标

- 基于当前双模型主线 `ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF LCC`，研究 ROI 方向是否还能提升 Precision 和 F1-Score。
- 当前主线问题不是漏诊过高，而是 FP 偏多：`TN 357 / FP 80 / FN 31 / TP 179`。
- 本轮只保留能带来明确收益的旁路候选，不直接覆盖 `configs/inference/demo.yml`。

## 当前主线

| 方案 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | 混淆矩阵 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 当前双模型 ROI OOF LCC | 0.9208 | 0.550 | 0.8524 | 0.8169 | 0.8284 | 0.6911 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |

## 已尝试方向

| 方向 | 代表配置 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | 决策 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 仅调阈值 | 当前概率，阈值 `0.560` | 0.9208 | 0.560 | 0.8429 | 0.8284 | 0.8331 | 0.7024 | 0.7662 | 放弃：提升太小 |
| 去掉 LCC | `demo_roi_oof.yml` | 0.9196 | 0.550 | 0.8429 | 0.8330 | 0.8362 | 0.7080 | 0.7696 | 放弃：AUC 和 Sensitivity 回退 |
| OOF F1 面积门控 | ROI 面积 `<0.25` 回退 full | 0.9092 | 0.520 | 0.8524 | 0.8284 | 0.8362 | 0.7047 | 0.7716 | 放弃：AUC 下降过大 |
| Pareto 面积门控 | ROI 面积 `<0.20` 或 `>0.95` 回退 full | 0.9141 | 0.520 | 0.8619 | 0.8398 | 0.8470 | 0.7211 | 0.7852 | 放弃：F1 提升但 AUC 下降 |
| F1 候选面积门控 | ROI 面积 `<0.08` 或 `>0.75` 回退 full | 0.9256 | 0.510 | 0.8667 | 0.8467 | 0.8532 | 0.7309 | 0.7930 | 保留：全指标优于当前主线 |

## 保留候选

保留配置：

- `configs/inference/demo_roi_area_gate_f1_candidate.yml`

核心逻辑：

- segmenter 生成 ROI mask；
- 使用 `mask_threshold=0.40`、`margin_ratio=0.35` 和最大连通域；
- 计算扩展后 ROI 面积占比；
- 如果 ROI 面积 `<0.08` 或 `>0.75`，认为 ROI 质量不稳定，回退到 full-image 概率；
- 否则继续使用原来的 ROI OOF stacker；
- 默认判定阈值改为 `0.510`。

## 收益

| 指标 | 当前主线 | 面积门控候选 | 变化 |
| --- | ---: | ---: | ---: |
| AUC | 0.9208 | 0.9256 | +0.0048 |
| Sensitivity | 0.8524 | 0.8667 | +0.0143 |
| Specificity | 0.8169 | 0.8467 | +0.0297 |
| Accuracy | 0.8284 | 0.8532 | +0.0247 |
| Precision | 0.6911 | 0.7309 | +0.0398 |
| F1-Score | 0.7633 | 0.7930 | +0.0297 |
| FP | 80 | 67 | -13 |
| FN | 31 | 28 | -3 |

## 判断

- ROI 面积门控是本轮最有效的 ROI 方向：它不是继续扩大 ROI，也不是继续降低阈值，而是在 ROI 质量异常时回退 full-image，减少不可靠 ROI 带来的良性误报。
- 当前候选已经同时提升 AUC、Sensitivity、Specificity、Accuracy、Precision 和 F1-Score，值得保留为下一版主线候选。
- 仍不建议立刻覆盖 `demo.yml`：下一步应在 BUSBRA OOF 或队内固定验证协议上复核面积阈值，再决定是否正式合并到默认 demo。
