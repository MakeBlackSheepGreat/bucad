# 比赛指标复查与重新评估报告

日期：2026-04-26

## 复查结论

本次复查发现：当前 AUC、Accuracy、Recall/Sensitivity、Specificity 的计算公式没有发现偏差，但此前报告输出不完整，缺少比赛要求中的 `Precision` 和 `F1-Score`。因此已从 `src/utils/metrics.py` 的统一指标函数开始补齐，并重新生成 BUSI 外部评估 JSON 与 threshold markdown。

需要强调：BUSBRA 用于训练、内部验证、OOF、模型选择、阈值选择和 ROI 参数选择；BUSI 仍是锁定外部评估集。本报告中的 BUSI threshold sweep 仅作运行点分析，不能把 BUSI 上扫描出的 best threshold 当作训练或调参依据。正式汇报应优先引用 demo 固定阈值 `0.55` 下的指标。

## 比赛要求指标对应关系

| 比赛指标 | 本项目字段 | 计算含义 | 本次状态 |
| --- | --- | --- | --- |
| AUC | `auc` | 基于 malignant probability 计算 ROC-AUC | 已复核并输出 |
| Accuracy | `accuracy` | `(TP + TN) / 全部样本` | 已复核并输出 |
| Recall/Sensitivity | `sensitivity` / `recall` | `TP / (TP + FN)`，恶性检出率 | 已复核并输出 |
| Precision | `precision` | `TP / (TP + FP)`，预测为恶性中的真实恶性比例 | 已补齐并输出 |
| Specificity | `specificity` | `TN / (TN + FP)`，良性排除能力 | 已复核并输出 |
| F1-Score | `f1_score` | `2 * Precision * Recall / (Precision + Recall)` | 已补齐并输出 |

## 重新评估配置

| 项目 | 内容 |
| --- | --- |
| Evaluation dataset | BUSI external evaluation only |
| Config | `configs/inference/demo.yml` |
| Output JSON | `artifacts/reports/competition_metrics_demo_eval.json` |
| Threshold report | `artifacts/reports/threshold_analysis_competition_metrics_demo_eval.md` |
| Sample count | `647` |
| Classifier member count | `10` |
| Main model family | ConvNeXt-Tiny + EfficientNetV2-S ensemble with ROI OOF LCC mainline |

## 固定 Demo 阈值结果：threshold 0.55

该表是当前 demo 主线配置的推荐汇报口径，因为阈值来自配置文件，不是本次在 BUSI 上重新调出来的。

| AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Threshold |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9208 | 0.8284 | 0.8524 | 0.6911 | 0.8169 | 0.7633 | 0.5500 |

| TN | FP | FN | TP |
| ---: | ---: | ---: | ---: |
| 357 | 80 | 31 | 179 |

## BUSI Youden 分析点：threshold 0.51

该表仅用于外部评估后的运行点分析，不作为模型训练、权重选择或正式阈值选择依据。

| AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Threshold | Youden J |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9208 | 0.8269 | 0.9000 | 0.6750 | 0.7918 | 0.7714 | 0.5100 | 0.6918 |

| TN | FP | FN | TP |
| ---: | ---: | ---: | ---: |
| 346 | 91 | 21 | 189 |

## 偏差修复记录

- `src/utils/metrics.py`：新增 `precision`、`recall` alias 和 `f1_score`，并继续保留原有 `sensitivity` 字段。
- `src/engine/inference.py`：threshold markdown 的默认阈值、best Youden 和 sweep 表全部补齐 Precision 与 F1-Score。
- `src/utils/document_reports.py`：DOCX/结构化报告的指标表补齐 Precision 与 F1-Score。
- `scripts/search_busi_ensemble_weights.py`：权重搜索复用统一 threshold metrics，避免手写指标遗漏。
- `tests/unit/test_metrics.py`、`tests/integration/test_busi_eval.py`：新增 Precision/F1 断言。

## 验证命令

```powershell
C:\Users\876762330\.conda\envs\BUCAD\python.exe -m pytest tests\unit\test_metrics.py tests\integration\test_busi_eval.py
C:\Users\876762330\.conda\envs\BUCAD\python.exe scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\competition_metrics_demo_eval.json
```

## 验证结果

- 指标测试：`6 passed in 9.80s`
- 完整测试：`59 passed in 11.57s`
- BUSI 外部评估：已完成，输出包含 AUC、Accuracy、Recall/Sensitivity、Precision、Specificity、F1-Score 和 confusion matrix。
- 结论：此前报告字段不完整，需要修正；修正后已满足比赛技术指标口径。
