# Descriptor-guided ROI Router OOF 实验

## 实验边界

- 本脚本只读取 BUSBRA OOF 预测缓存、BUSBRA 图像，以及由 `descriptor_mask_source` 指定的内部 mask 口径，不读取 BUSI。
- Router 使用原始 fold_id 做嵌套 OOF：每个样本的 router 概率来自未见过该 fold 的轻量逻辑回归模型。
- 生成的 `demo_descriptor_router.yml` 是新分支实验配置，不等同于主线 `demo.yml`。

## 入选 Router

- 候选：`logistic_descriptor_full_C0.03_balanced`
- 特征集：`descriptor_full`
- 参数：`{"C": 0.03, "class_weight": "balanced"}`
- 是否满足内部合入门槛：`True`
- 描述符 mask 来源：`segmenter_oof`
- 实验配置：`artifacts\reports\Chinese reports\09_unetplusplus_resnet34_custom_roi\frozen_configs\demo_unetplusplus_resnet34_descriptor_router_relaxed.yml`
- 实验配置中 router 启用状态：`True`
- 内部门槛：AUC 至少提升 `0.0030`，F1 至少提升 `-0.0050`，Sensitivity/Specificity 最大下降 `0.0200`

使用特征：

- `full_probability`
- `roi_probability`
- `stacked_probability`
- `full_logit`
- `roi_logit`
- `stacked_logit`
- `full_roi_delta`
- `abs_full_roi_delta`
- `full_uncertainty`
- `roi_uncertainty`
- `stacked_uncertainty`
- `roi_valid`
- `roi_area_ratio`
- `mask_area_ratio`
- `lesion_bbox_area_ratio`
- `lesion_aspect_ratio`
- `mask_extent`
- `mask_compactness`
- `boundary_complexity`
- `component_count`
- `mask_mean_probability`
- `edge_contrast`
- `image_mean`
- `image_std`
- `image_sharpness`

## OOF 指标对比

| 方案 | AUC | 阈值 | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Youden J | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 当前 ROI stacker（demo 阈值） | 0.9234 | 0.570 | 0.8544 | 0.8402 | 0.8612 | 0.7434 | 0.7889 | 0.7014 | TN 1092 / FP 176 / FN 97 / TP 510 |
| 当前 ROI stacker（OOF Youden 阈值） | 0.9234 | 0.600 | 0.8635 | 0.8237 | 0.8825 | 0.7704 | 0.7962 | 0.7062 | TN 1119 / FP 149 / FN 107 / TP 500 |
| Descriptor router（嵌套 OOF） | 0.9271 | 0.520 | 0.8581 | 0.8402 | 0.8667 | 0.7511 | 0.7932 | 0.7069 | TN 1099 / FP 169 / FN 97 / TP 510 |

## Top 候选

| 排名 | 候选 | 特征集 | AUC | 阈值 | Sensitivity | Specificity | Precision | F1-Score | Confusion |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `logistic_descriptor_full_C0.03_balanced` | `descriptor_full` | 0.9271 | 0.520 | 0.8402 | 0.8667 | 0.7511 | 0.7932 | TN 1099 / FP 169 / FN 97 / TP 510 |
| 2 | `logistic_descriptor_full_C0.01_none` | `descriptor_full` | 0.9270 | 0.370 | 0.8287 | 0.8722 | 0.7564 | 0.7909 | TN 1106 / FP 162 / FN 104 / TP 503 |
| 3 | `logistic_descriptor_full_C0.03_none` | `descriptor_full` | 0.9270 | 0.350 | 0.8369 | 0.8636 | 0.7460 | 0.7888 | TN 1095 / FP 173 / FN 99 / TP 508 |
| 4 | `logistic_descriptor_full_C0.1_balanced` | `descriptor_full` | 0.9268 | 0.530 | 0.8303 | 0.8715 | 0.7556 | 0.7912 | TN 1105 / FP 163 / FN 103 / TP 504 |
| 5 | `logistic_descriptor_full_C0.01_balanced` | `descriptor_full` | 0.9268 | 0.490 | 0.8534 | 0.8533 | 0.7358 | 0.7902 | TN 1082 / FP 186 / FN 89 / TP 518 |
| 6 | `logistic_descriptor_full_C0.1_none` | `descriptor_full` | 0.9267 | 0.290 | 0.8616 | 0.8352 | 0.7145 | 0.7812 | TN 1059 / FP 209 / FN 84 / TP 523 |
| 7 | `logistic_descriptor_full_C0.3_balanced` | `descriptor_full` | 0.9264 | 0.490 | 0.8468 | 0.8525 | 0.7332 | 0.7859 | TN 1081 / FP 187 / FN 93 / TP 514 |
| 8 | `logistic_descriptor_full_C0.3_none` | `descriptor_full` | 0.9263 | 0.360 | 0.8270 | 0.8699 | 0.7526 | 0.7881 | TN 1103 / FP 165 / FN 105 / TP 502 |
| 9 | `logistic_descriptor_full_C1_balanced` | `descriptor_full` | 0.9259 | 0.490 | 0.8501 | 0.8525 | 0.7340 | 0.7878 | TN 1081 / FP 187 / FN 91 / TP 516 |
| 10 | `logistic_descriptor_full_C1_none` | `descriptor_full` | 0.9259 | 0.300 | 0.8567 | 0.8375 | 0.7163 | 0.7802 | TN 1062 / FP 206 / FN 87 / TP 520 |

## 结论

- 建议：`candidate_for_frozen_external_review`
- 若内部 OOF 未明显超过当前 ROI stacker，实验配置会写入 `descriptor_router.enabled: false`，避免误用。
- 若后续要继续推进，下一步应在固定 router 配置后只做一次 BUSI 外部复核，不能用 BUSI 继续反向调参。
