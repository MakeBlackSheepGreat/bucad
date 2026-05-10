# 比赛指标报告回归检查

## 检查背景

用户复核后指出：README 顶部评估结果和一部分历史实验报告仍沿用旧指标表，只展示 AUC、Sensitivity、Specificity、Accuracy，缺少比赛要求中的 `Precision` 和 `F1-Score`。本次回归检查的目标是统一报告口径，而不是重新训练模型、重新选择模型或重新调参。

## 比赛指标标准

后续 BUSI 外部评估运行点必须统一展示以下指标：

| 指标 | 含义 | 当前报告字段 |
| --- | --- | --- |
| AUC | 曲线下面积，衡量模型区分良恶性样本的整体能力 | `auc` |
| Accuracy | 预测正确样本占总样本比例 | `accuracy` |
| Recall/Sensitivity | 实际恶性样本中被正确识别为恶性的比例 | `sensitivity` / `recall` |
| Precision | 被预测为恶性的样本中真实恶性的比例 | `precision` |
| Specificity | 实际良性样本中被正确识别为良性的比例 | `specificity` |
| F1-Score | Precision 与 Recall/Sensitivity 的调和平均 | `f1_score` |

## 本次修正范围

- README 顶部 demo 主线结果表已补齐 `Precision` 和 `F1-Score`，并改为完整比赛指标顺序。
- README 顶部三模型非 ROI 对照表已补齐 `Precision` 和 `F1-Score`。
- 顶层实验报告中带有 `TN / FP / FN / TP` 混淆矩阵的表格，已根据混淆矩阵自动补齐 `Precision` 和 `F1-Score`。
- 归档英文/中文阈值分析报告已基于对应 BUSI JSON 重新生成，默认阈值、Youden J 最优阈值和阈值扫描表均包含完整比赛指标。
- 对于没有保存混淆矩阵或原始概率的历史摘要表，报告中保留 `-`，并追加“比赛指标完整性说明”，避免伪造无法反推的指标。

## 数据边界声明

本次只修正报告展示字段，不改变任何模型权重、阈值、ROI 参数或集成权重。BUSBRA 仍用于训练、内部验证、OOF 选择、ROI 参数选择和阈值选择；BUSI 仍只用于配置锁定后的外部评估，不参与训练或调参。

## 当前主线完整指标

| 配置 | AUC | 阈值 | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF Stacking + LCC ROI | 0.9208 | 0.550 | 0.8284 | 0.8524 | 0.6911 | 0.8169 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |

## 完整指标来源

- `artifacts/reports/competition_metrics_all_busi_reports.md`：所有 BUSI JSON 报告的完整指标汇总。
- `artifacts/reports/competition_metrics_audit.md`：指标公式和评估字段审计。
- `artifacts/reports/threshold_analysis.md`：当前主线阈值扫描。

