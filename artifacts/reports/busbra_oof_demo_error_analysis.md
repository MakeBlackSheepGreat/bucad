# BUSBRA OOF Demo 主线错误样本分析

## 数据边界

- 内部分析使用 BUSBRA OOF 预测缓存：每个样本由对应折外的分类器预测，适合作为训练集内部测试/误差分析口径。
- ROI OOF 缓存使用 BUSBRA 真值 mask 生成 ROI，再跑同一组分类器；因此它分析的是当前 ROI 融合/面积门逻辑，不等同于分割器预测 mask 的端到端内部成绩。
- BUSI 只在本报告末尾作为锁定外部验证结果列出，不参与阈值、gate、模型或样本选择。

## 内部 OOF 当前 demo 运行点

- 配置：`configs/inference/demo.yml`
- 阈值：`0.51`
- AUC：`0.9166`，Sensitivity：`0.8402`，Specificity：`0.8423`，Precision：`0.7183`，F1：`0.7745`
- 混淆矩阵：TN `1068`，FP `200`，FN `97`，TP `510`
- 错误样本：`297` 个，其中 FP `200` 个，FN `97` 个。

## 错误机制

| 类别 | 数量 | 占比 |
| --- | ---: | ---: |
| roi_stacker_pushed_to_malignant | 120 | 40.4% |
| full_image_error_area_gate_fallback | 63 | 21.2% |
| full_image_already_benign | 59 | 19.9% |
| full_image_already_malignant | 55 | 18.5% |

### FP 机制

| 类别 | 数量 | 占比 |
| --- | ---: | ---: |
| roi_stacker_pushed_to_malignant | 120 | 60.0% |
| full_image_already_malignant | 55 | 27.5% |
| full_image_error_area_gate_fallback | 25 | 12.5% |

### FN 机制

| 类别 | 数量 | 占比 |
| --- | ---: | ---: |
| full_image_already_benign | 59 | 60.8% |
| full_image_error_area_gate_fallback | 38 | 39.2% |

## 置信度与面积

### 置信度分布

| 类别 | 数量 | 占比 |
| --- | ---: | ---: |
| confident | 218 | 73.4% |
| borderline | 54 | 18.2% |
| near_threshold | 25 | 8.4% |

### ROI 面积分布

| 类别 | 数量 | 占比 |
| --- | ---: | ---: |
| medium | 93 | 31.3% |
| large | 77 | 25.9% |
| small | 64 | 21.5% |
| very_large | 58 | 19.5% |
| small_or_empty | 5 | 1.7% |

### 分组均值

| 错误类型 | final均值 | full均值 | ROI均值 | ROI面积均值 | GT mask面积均值 | 图像均值 | 图像std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP | 0.6659 | 0.4892 | 0.5882 | 0.4425 | 0.0880 | 0.3254 | 0.1528 |
| FN | 0.2570 | 0.1412 | 0.2981 | 0.5530 | 0.1133 | 0.3161 | 0.1473 |

## 分支错因定位

| 错误类型 | full_eff | full_conv | roi_eff | roi_conv | full_pair | roi_pair | gate_fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP | 48.5% | 50.0% | 42.5% | 76.0% | 40.0% | 60.5% | 12.5% |
| FN | 84.5% | 92.8% | 75.3% | 73.2% | 100.0% | 77.3% | 39.2% |

## 元数据分布

### BIRADS

| birads | FP | FN |
| --- | ---: | ---: |
| 2 | 44 | 2 |
| 3 | 56 | 18 |
| 4 | 97 | 72 |
| 5 | 3 | 5 |

### view_side

| view_side | FP | FN |
| --- | ---: | ---: |
| left | 88 | 42 |
| right | 84 | 42 |
| single | 28 | 13 |

## 代表性内部错误样本

### 最高置信 FP

| sample_id | fold | p(final) | full | ROI | ROI面积 | BIRADS | 机制 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bus_0963-l | 3 | 1.000 | 1.000 | 1.000 | 0.784 | 4 | full_image_error_area_gate_fallback |
| bus_0962-r | 4 | 0.997 | 0.997 | 0.956 | 0.771 | 4 | full_image_error_area_gate_fallback |
| bus_0270-l | 1 | 0.985 | 0.999 | 0.960 | 0.519 | 4 | full_image_already_malignant |
| bus_1038-l | 4 | 0.966 | 0.966 | 0.960 | 0.807 | 4 | full_image_error_area_gate_fallback |
| bus_1038-r | 4 | 0.954 | 0.993 | 0.885 | 0.584 | 4 | full_image_already_malignant |
| bus_0270-r | 1 | 0.953 | 0.953 | 0.916 | 0.877 | 4 | full_image_error_area_gate_fallback |
| bus_0385-r | 3 | 0.945 | 0.760 | 0.999 | 0.498 | 4 | full_image_already_malignant |
| bus_0963-r | 3 | 0.942 | 0.942 | 0.944 | 0.793 | 4 | full_image_error_area_gate_fallback |
| bus_0427-l | 2 | 0.939 | 0.982 | 0.914 | 0.553 | 5 | full_image_already_malignant |
| bus_0359-r | 3 | 0.929 | 0.829 | 0.997 | 0.664 | 4 | full_image_already_malignant |

### 最高置信 FN

| sample_id | fold | p(final) | full | ROI | ROI面积 | BIRADS | 机制 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bus_0298-r | 5 | 0.000 | 0.000 | 0.219 | 0.075 | 3 | full_image_error_area_gate_fallback |
| bus_0190-l | 1 | 0.001 | 0.001 | 0.600 | 0.060 | 4 | full_image_error_area_gate_fallback |
| bus_0862-l | 4 | 0.003 | 0.003 | 0.000 | 0.799 | 4 | full_image_error_area_gate_fallback |
| bus_0231-l | 3 | 0.004 | 0.000 | 0.000 | 0.689 | 4 | full_image_already_benign |
| bus_0238-r | 1 | 0.007 | 0.007 | 0.097 | 0.859 | 4 | full_image_error_area_gate_fallback |
| bus_0334-l | 4 | 0.010 | 0.010 | 0.198 | 0.902 | 3 | full_image_error_area_gate_fallback |
| bus_0287-l | 5 | 0.017 | 0.017 | 0.201 | 0.888 | 5 | full_image_error_area_gate_fallback |
| bus_0012-r | 4 | 0.018 | 0.000 | 0.019 | 0.388 | 4 | full_image_already_benign |
| bus_0052-l | 3 | 0.018 | 0.018 | 0.017 | 0.924 | 4 | full_image_error_area_gate_fallback |
| bus_0298-l | 5 | 0.019 | 0.000 | 0.016 | 0.102 | 3 | full_image_already_benign |

## BUSI 锁定外部验证

- BUSI 配置：`configs/inference/demo.yml`
- AUC：`0.9256`，Sensitivity：`0.8667`，Specificity：`0.8467`，Precision：`0.7309`，F1：`0.7930`
- 混淆矩阵：TN `370`，FP `67`，FN `28`，TP `182`
- 该外部验证只用于确认锁定配置的泛化表现，不用于本轮错误样本选择或参数调整。

## 结论

- 内部 OOF 下，FP 中 `120` 个是 ROI/stacker 把良性推过阈值，`25` 个是面积门回退后完整图误报。
- 内部 OOF 下，FN 中 `59` 个在完整图分支已经偏良性，优先说明分类表征/训练样本覆盖问题，而不是单纯 ROI 后处理问题。
- 后续优化应只基于 BUSBRA OOF 的这些内部错误模式做 hard-case mining、增强和融合器重训；完成后再对 BUSI 做一次锁定外部复核。
