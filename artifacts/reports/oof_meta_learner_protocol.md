# OOF Meta-Learner ROI 融合实验

日期：2026-04-26

## 实验边界

- 只使用 BUSBRA OOF 预测、ROI 面积和标签进行候选选择。
- 使用原 5 折 fold_id 做嵌套 OOF：每个样本的 meta 预测来自未见过该 fold 的 meta 模型。
- 外部数据只在 OOF 选定模型、阈值和特征后做一次固定复核，不参与训练或调参。

## 入选方案

- 候选名称：`logistic_logit_rich_C0.03_none`
- 模型族：`logistic_regression`
- 特征集：`logit_rich`
- 参数：`{"C": 0.03, "class_weight": null}`
- OOF 阈值：`0.370`

使用特征：

- `logit_runtime_stack`
- `logit_full_blend`
- `logit_roi_blend`
- `logit_full_eff`
- `logit_full_conv`
- `logit_roi_eff`
- `logit_roi_conv`
- `area_ratio`
- `log_area`
- `sqrt_area`
- `abs_full_roi_delta`
- `abs_full_model_delta`
- `abs_roi_model_delta`
- `area_bin_small`
- `area_bin_medium`
- `area_bin_large`
- `area_bin_very_large`

## 完整指标对比

| 方案 | 样本数 | 阳性 | 阴性 | AUC | 阈值 | Accuracy | Sensitivity | Recall | Specificity | Precision | NPV | F1-Score | Balanced Acc | Youden J | FPR | FNR | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OOF 当前主线 | 1875 | 607 | 1268 | 0.9239 | 0.550 | 0.8459 | 0.8534 | 0.8534 | 0.8423 | 0.7214 | 0.9231 | 0.7819 | 0.8478 | 0.6956 | 0.1577 | 0.1466 | TN 1068 / FP 200 / FN 89 / TP 518 |
| OOF meta-learner | 1875 | 607 | 1268 | 0.9241 | 0.370 | 0.8581 | 0.8468 | 0.8468 | 0.8636 | 0.7482 | 0.9217 | 0.7944 | 0.8552 | 0.7104 | 0.1364 | 0.1532 | TN 1095 / FP 173 / FN 93 / TP 514 |
| 外部当前主线 | 647 | 210 | 437 | 0.9208 | 0.550 | 0.8284 | 0.8524 | 0.8524 | 0.8169 | 0.6911 | 0.9201 | 0.7633 | 0.8347 | 0.6693 | 0.1831 | 0.1476 | TN 357 / FP 80 / FN 31 / TP 179 |
| 外部 meta-learner 固定复核 | 647 | 210 | 437 | 0.9115 | 0.370 | 0.8176 | 0.8714 | 0.8714 | 0.7918 | 0.6679 | 0.9276 | 0.7562 | 0.8316 | 0.6632 | 0.2082 | 0.1286 | TN 346 / FP 91 / FN 27 / TP 183 |

## OOF Top 候选

| 排名 | 候选 | 模型族 | 特征集 | AUC | 阈值 | Accuracy | Sensitivity | Specificity | Precision | F1-Score | FPR | FNR | Confusion |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `logistic_logit_rich_C0.03_none` | `logistic_regression` | `logit_rich` | 0.9241 | 0.370 | 0.8581 | 0.8468 | 0.8636 | 0.7482 | 0.7944 | 0.1364 | 0.1532 | TN 1095 / FP 173 / FN 93 / TP 514 |
| 2 | `logistic_logit_rich_C0.1_none` | `logistic_regression` | `logit_rich` | 0.9240 | 0.360 | 0.8560 | 0.8418 | 0.8628 | 0.7460 | 0.7910 | 0.1372 | 0.1582 | TN 1094 / FP 174 / FN 96 / TP 511 |
| 3 | `logistic_logit_rich_C0.03_balanced` | `logistic_regression` | `logit_rich` | 0.9237 | 0.530 | 0.8544 | 0.8484 | 0.8573 | 0.7399 | 0.7905 | 0.1427 | 0.1516 | TN 1087 / FP 181 / FN 92 / TP 515 |
| 4 | `logistic_logit_rich_C0.1_balanced` | `logistic_regression` | `logit_rich` | 0.9235 | 0.520 | 0.8533 | 0.8484 | 0.8557 | 0.7378 | 0.7893 | 0.1443 | 0.1516 | TN 1085 / FP 183 / FN 92 / TP 515 |
| 5 | `logistic_logit_rich_C0.3_none` | `logistic_regression` | `logit_rich` | 0.9238 | 0.350 | 0.8539 | 0.8435 | 0.8588 | 0.7410 | 0.7889 | 0.1412 | 0.1565 | TN 1089 / FP 179 / FN 95 / TP 512 |
| 6 | `logistic_logit_rich_C0.3_balanced` | `logistic_regression` | `logit_rich` | 0.9232 | 0.530 | 0.8523 | 0.8402 | 0.8580 | 0.7391 | 0.7864 | 0.1420 | 0.1598 | TN 1088 / FP 180 / FN 97 / TP 510 |
| 7 | `logistic_logit_rich_C1_none` | `logistic_regression` | `logit_rich` | 0.9236 | 0.340 | 0.8507 | 0.8435 | 0.8541 | 0.7346 | 0.7853 | 0.1459 | 0.1565 | TN 1083 / FP 185 / FN 95 / TP 512 |
| 8 | `logistic_logit_rich_C3_none` | `logistic_regression` | `logit_rich` | 0.9234 | 0.330 | 0.8491 | 0.8484 | 0.8494 | 0.7295 | 0.7845 | 0.1506 | 0.1516 | TN 1077 / FP 191 / FN 92 / TP 515 |
| 9 | `hgb_probability_rich_lr0.03_leaf3_l20.1_iter120` | `hist_gradient_boosting` | `probability_rich` | 0.9186 | 0.300 | 0.8517 | 0.8616 | 0.8470 | 0.7294 | 0.7900 | 0.1530 | 0.1384 | TN 1074 / FP 194 / FN 84 / TP 523 |
| 10 | `hgb_probability_rich_lr0.05_leaf3_l21_iter80` | `hist_gradient_boosting` | `probability_rich` | 0.9187 | 0.320 | 0.8528 | 0.8550 | 0.8517 | 0.7341 | 0.7900 | 0.1483 | 0.1450 | TN 1080 / FP 188 / FN 88 / TP 519 |

## 结论

- OOF meta-learner 未形成稳定综合提升，建议暂不合并主线。
- 当前实验已经列出 AUC、Accuracy、Recall/Sensitivity、Precision、Specificity、F1-Score，以及 NPV、Balanced Accuracy、Youden J、FPR、FNR 和混淆矩阵。
