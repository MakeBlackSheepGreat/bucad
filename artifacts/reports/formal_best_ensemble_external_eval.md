# 两模型与三模型最优集成正式外部评估

> 数据边界说明：BUSBRA 用于训练、内部验证、OOF、模型选择、阈值选择和 ROI 参数选择；BUSI 是锁定外部评估集，只用于最终外部验证，不用于训练或调参。


日期：2026-04-25

## 评估说明

- 数据集：BUSI 外部评估集，`647` 张非 normal 样本。
- 评估方式：直接加载本地 checkpoint，对 BUSI 全量图片重新执行推理，不使用缓存概率重算。
- 推理修正：本次已支持成员级预处理与 TTA；EfficientNetV2-S、DenseNet121、ConvNeXt-Tiny 分别使用各自训练/评估配置。
- 数据边界：BUSI 仅用于外部评估和阈值分析，不参与训练。

## 配置与权重

| 方案 | 配置文件 | 成员数量 | 权重 |
| --- | --- | ---: | --- |
| 两模型 AUC 最优 | `configs/inference/ensemble_effnet_convnext_auc_best.yml` | 10 | EfficientNetV2-S `0.427` / ConvNeXt-Tiny `0.573` |
| 三模型 AUC 最优 | `configs/inference/ensemble_effnet_densenet_convnext_auc_best.yml` | 15 | EfficientNetV2-S `0.358` / DenseNet121 `0.244` / ConvNeXt-Tiny `0.398` |

## 正式评估结果

| 方案 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | 混淆矩阵 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 两模型，默认阈值 | 0.9142 | 0.50 | 0.7524 | 0.9130 | 0.8609 | 0.8061 | 0.7783 | TN 399 / FP 38 / FN 52 / TP 158 |
| 两模型，Youden 最优阈值 | 0.9142 | 0.47 | 0.7714 | 0.9108 | 0.8655 | 0.8060 | 0.7883 | TN 398 / FP 39 / FN 48 / TP 162 |
| 三模型，默认阈值 | 0.9144 | 0.50 | 0.7095 | 0.9382 | 0.8640 | 0.8466 | 0.7720 | TN 410 / FP 27 / FN 61 / TP 149 |
| 三模型，Youden 最优阈值 | 0.9144 | 0.35 | 0.7952 | 0.8787 | 0.8516 | 0.7591 | 0.7767 | TN 384 / FP 53 / FN 43 / TP 167 |

## 结论

- 三模型正式 AUC 略高：`0.9144`，比两模型 `0.9142` 高 `0.0001`，差距非常小。
- 两模型在 Youden 最优点的 Accuracy 更高：`0.8655`，且模型数量更少，部署成本更低。
- 三模型默认阈值更保守：Specificity `0.9382`，误报更少，但 Sensitivity 只有 `0.7095`。
- 如果比赛展示优先看 AUC，可以保留三模型作为最终最优结果；如果重视部署简洁和综合运行点，两模型方案更实用。

## 输出文件

- 两模型 JSON：`artifacts/reports/busi_ensemble_effnet_convnext_auc_best_formal.json`
- 两模型阈值分析：`artifacts/reports/threshold_analysis_busi_ensemble_effnet_convnext_auc_best_formal.md`
- 三模型 JSON：`artifacts/reports/busi_ensemble_effnet_densenet_convnext_auc_best_formal.json`
- 三模型阈值分析：`artifacts/reports/threshold_analysis_busi_ensemble_effnet_densenet_convnext_auc_best_formal.md`
