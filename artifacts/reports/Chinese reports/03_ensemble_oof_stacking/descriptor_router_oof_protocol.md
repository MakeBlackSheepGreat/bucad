# Descriptor-guided ROI Router OOF 实验

## 实验边界

- 本脚本只读取 BUSBRA OOF 预测缓存、BUSBRA 图像，以及由 `descriptor_mask_source` 指定的内部 mask 口径，不读取 BUSI。
- Router 使用原始 fold_id 做嵌套 OOF：每个样本的 router 概率来自未见过该 fold 的轻量逻辑回归模型。
- 生成的 `demo_descriptor_router.yml` 是新分支实验配置，不等同于主线 `demo.yml`。

## 入选 Router

- 候选：`logistic_descriptor_full_C0.1_none`
- 特征集：`descriptor_full`
- 参数：`{"C": 0.1, "class_weight": null}`
- 是否满足内部合入门槛：`False`
- 描述符 mask 来源：`segmenter_oof`
- 实验配置：`configs/inference/demo_descriptor_router.yml`
- 实验配置中 router 启用状态：`False`
- 内部门槛：AUC 至少提升 `0.0050`，F1 至少提升 `0.0050`，Sensitivity/Specificity 最大下降 `0.0100`

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
| 当前 ROI stacker（demo 阈值） | 0.9158 | 0.510 | 0.8427 | 0.8501 | 0.8391 | 0.7167 | 0.7777 | 0.6892 | TN 1064 / FP 204 / FN 91 / TP 516 |
| 当前 ROI stacker（OOF Youden 阈值） | 0.9158 | 0.570 | 0.8581 | 0.8138 | 0.8793 | 0.7635 | 0.7879 | 0.6932 | TN 1115 / FP 153 / FN 113 / TP 494 |
| Descriptor router（嵌套 OOF） | 0.9175 | 0.370 | 0.8587 | 0.8303 | 0.8722 | 0.7568 | 0.7918 | 0.7026 | TN 1106 / FP 162 / FN 103 / TP 504 |

## Top 候选

| 排名 | 候选 | 特征集 | AUC | 阈值 | Sensitivity | Specificity | Precision | F1-Score | Confusion |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `logistic_descriptor_full_C0.1_none` | `descriptor_full` | 0.9175 | 0.370 | 0.8303 | 0.8722 | 0.7568 | 0.7918 | TN 1106 / FP 162 / FN 103 / TP 504 |
| 2 | `logistic_descriptor_full_C0.3_none` | `descriptor_full` | 0.9174 | 0.370 | 0.8303 | 0.8722 | 0.7568 | 0.7918 | TN 1106 / FP 162 / FN 103 / TP 504 |
| 3 | `logistic_descriptor_full_C1_none` | `descriptor_full` | 0.9172 | 0.400 | 0.8188 | 0.8825 | 0.7693 | 0.7933 | TN 1119 / FP 149 / FN 110 / TP 497 |
| 4 | `logistic_descriptor_full_C0.03_none` | `descriptor_full` | 0.9172 | 0.370 | 0.8237 | 0.8738 | 0.7576 | 0.7893 | TN 1108 / FP 160 / FN 107 / TP 500 |
| 5 | `logistic_descriptor_full_C3_none` | `descriptor_full` | 0.9170 | 0.390 | 0.8204 | 0.8785 | 0.7638 | 0.7911 | TN 1114 / FP 154 / FN 109 / TP 498 |
| 6 | `logistic_descriptor_boundary_quality_C0.3_none` | `descriptor_boundary_quality` | 0.9167 | 0.380 | 0.8237 | 0.8628 | 0.7418 | 0.7806 | TN 1094 / FP 174 / FN 107 / TP 500 |
| 7 | `logistic_descriptor_boundary_quality_C1_none` | `descriptor_boundary_quality` | 0.9165 | 0.370 | 0.8303 | 0.8580 | 0.7368 | 0.7808 | TN 1088 / FP 180 / FN 103 / TP 504 |
| 8 | `logistic_descriptor_boundary_quality_C0.1_none` | `descriptor_boundary_quality` | 0.9165 | 0.370 | 0.8287 | 0.8612 | 0.7408 | 0.7823 | TN 1092 / FP 176 / FN 104 / TP 503 |
| 9 | `logistic_descriptor_boundary_quality_C3_none` | `descriptor_boundary_quality` | 0.9164 | 0.370 | 0.8303 | 0.8565 | 0.7347 | 0.7796 | TN 1086 / FP 182 / FN 103 / TP 504 |
| 10 | `logistic_descriptor_full_C0.01_none` | `descriptor_full` | 0.9160 | 0.370 | 0.8254 | 0.8707 | 0.7534 | 0.7877 | TN 1104 / FP 164 / FN 106 / TP 501 |

## 结论

- 建议：`do_not_merge_without_more_internal_evidence`
- 若内部 OOF 未明显超过当前 ROI stacker，实验配置会写入 `descriptor_router.enabled: false`，避免误用。
- 若后续要继续推进，下一步应在固定 router 配置后只做一次 BUSI 外部复核，不能用 BUSI 继续反向调参。

## 冻结后外部复核

为确认修正后的 `segmenter_oof` 描述符口径是否具备迁移价值，固定内部候选后执行一次外部复核：

```powershell
conda run -n BUCAD python scripts\eval_busi.py --config configs\inference\demo_descriptor_router.yml --output artifacts\reports\busi_demo_descriptor_router_segmenter_oof_frozen_review.json
```

| 方案 | AUC | 阈值 | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 当前主线 `demo.yml` | 0.9256 | 0.510 | 0.8532 | 0.8667 | 0.8467 | 0.7309 | 0.7930 | TN 370 / FP 67 / FN 28 / TP 182 |
| 旧版真值 mask descriptor router | 0.8727 | 0.360 | 0.7728 | 0.8476 | 0.7368 | 0.6075 | 0.7078 | TN 322 / FP 115 / FN 32 / TP 178 |
| 修正后 `segmenter_oof` descriptor router | 0.9071 | 0.370 | 0.8423 | 0.8000 | 0.8627 | 0.7368 | 0.7671 | TN 377 / FP 60 / FN 42 / TP 168 |

修正后外部 AUC 从 `0.8727` 回升到 `0.9071`，说明真值 mask 描述符泄漏问题已缓解；但它仍低于当前主线 `0.9256`，且 sensitivity 与 F1 均下降。因此当前解决策略是：保留 descriptor router 的工程实现和 OOF 协议，实验配置中将 `descriptor_router.enabled` 写为 `false`，不合入主线 `demo.yml`。
