# BUSI Demo 错误样本分析

## 数据边界

- BUSI 是锁定外部测试/评估集，本报告只能用于事后错误归因和答辩说明。
- 禁止用本报告中的 BUSI 样本、错误类型、阈值距离或 ROI 面积分布来调阈值、调 ROI gate、选择分割器、选择模型、训练 hard negative 或决定合入。
- 任何性能优化都必须回到 BUSBRA 的训练/验证/OOF 流程内完成；BUSI 只在配置锁定后做外部复核。

## 当前运行点

- 配置：`configs\inference\demo.yml`
- 阈值：`0.51`
- AUC：`0.9256`，Sensitivity：`0.8667`，Specificity：`0.8467`，Precision：`0.7309`，F1：`0.7930`
- 混淆矩阵：TN `370`，FP `67`，FN `28`，TP `182`
- 错误样本：`95` 个，其中 FP `67` 个，FN `28` 个。
- 主线概率重建最大误差：`0.000089`；该误差只来自缓存/浮点差异。

## 主要错误机制

| 类别 | 数量 | 占比 |
| --- | ---: | ---: |
| full_image_error_area_gate_fallback | 32 | 33.7% |
| roi_stacker_pushed_to_malignant | 31 | 32.6% |
| full_image_already_benign | 20 | 21.1% |
| full_image_already_malignant | 12 | 12.6% |

### FP 机制

| 类别 | 数量 | 占比 |
| --- | ---: | ---: |
| roi_stacker_pushed_to_malignant | 31 | 46.3% |
| full_image_error_area_gate_fallback | 24 | 35.8% |
| full_image_already_malignant | 12 | 17.9% |

### FN 机制

| 类别 | 数量 | 占比 |
| --- | ---: | ---: |
| full_image_already_benign | 20 | 71.4% |
| full_image_error_area_gate_fallback | 8 | 28.6% |

## 置信度与 ROI 面积

### 置信度分布

| 类别 | 数量 | 占比 |
| --- | ---: | ---: |
| confident | 65 | 68.4% |
| borderline | 17 | 17.9% |
| near_threshold | 13 | 13.7% |

### ROI 面积分布

| 类别 | 数量 | 占比 |
| --- | ---: | ---: |
| very_large | 31 | 32.6% |
| large | 28 | 29.5% |
| medium | 22 | 23.2% |
| small | 13 | 13.7% |
| small_or_empty | 1 | 1.1% |

### 分组均值

| 错误类型 | final均值 | full均值 | ROI均值 | ROI面积均值 | GT mask面积均值 | 图像均值 | 图像std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP | 0.6566 | 0.5267 | 0.5921 | 0.6708 | 0.1192 | 0.3307 | 0.2048 |
| FN | 0.3318 | 0.1752 | 0.2242 | 0.4341 | 0.0718 | 0.3573 | 0.2065 |

## 分支错因定位

| 错误类型 | full_eff | full_conv | roi_eff | roi_conv | full_pair | roi_pair | gate_fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP | 35.8% | 59.7% | 50.7% | 77.6% | 53.7% | 71.6% | 35.8% |
| FN | 96.4% | 96.4% | 96.4% | 89.3% | 100.0% | 100.0% | 28.6% |

## 代表性错误样本

### 最高置信 FP

| sample_id | p(final) | full | ROI | 面积 | 机制 |
| --- | ---: | ---: | ---: | ---: | --- |
| benign (409) | 0.976 | 0.976 | 0.976 | 1.000 | full_image_error_area_gate_fallback |
| benign (397) | 0.958 | 0.958 | 0.958 | 1.000 | full_image_error_area_gate_fallback |
| benign (401) | 0.925 | 0.843 | 0.995 | 0.429 | full_image_already_malignant |
| benign (408) | 0.845 | 0.845 | 0.845 | 1.000 | full_image_error_area_gate_fallback |
| benign (42) | 0.837 | 0.830 | 0.836 | 0.679 | full_image_already_malignant |
| benign (400) | 0.835 | 0.835 | 0.835 | 1.000 | full_image_error_area_gate_fallback |
| benign (131) | 0.834 | 0.838 | 0.811 | 0.678 | full_image_already_malignant |
| benign (410) | 0.813 | 0.813 | 0.813 | 1.000 | full_image_error_area_gate_fallback |
| benign (407) | 0.812 | 0.812 | 0.812 | 1.000 | full_image_error_area_gate_fallback |
| benign (412) | 0.808 | 0.808 | 0.808 | 1.000 | full_image_error_area_gate_fallback |

### 最高置信 FN

| sample_id | p(final) | full | ROI | 面积 | 机制 |
| --- | ---: | ---: | ---: | ---: | --- |
| malignant (10) | 0.088 | 0.007 | 0.011 | 0.124 | full_image_already_benign |
| malignant (112) | 0.095 | 0.006 | 0.018 | 0.124 | full_image_already_benign |
| malignant (26) | 0.154 | 0.013 | 0.055 | 0.253 | full_image_already_benign |
| malignant (200) | 0.160 | 0.017 | 0.041 | 0.227 | full_image_already_benign |
| malignant (27) | 0.169 | 0.031 | 0.020 | 0.227 | full_image_already_benign |
| malignant (121) | 0.203 | 0.011 | 0.224 | 0.113 | full_image_already_benign |
| malignant (15) | 0.256 | 0.256 | 0.223 | 0.927 | full_image_error_area_gate_fallback |
| malignant (73) | 0.280 | 0.079 | 0.061 | 0.176 | full_image_already_benign |
| malignant (148) | 0.285 | 0.038 | 0.208 | 0.462 | full_image_already_benign |
| malignant (28) | 0.295 | 0.033 | 0.290 | 0.141 | full_image_already_benign |

## 结论

- FN 不是简单阈值问题：`22/28` 个 FN 离阈值超过 0.08，且 full/ROI 分支大多已经把恶性压低。
- FP 主要分两类：ROI/stacker 把良性推到恶性的 `31` 个，以及 ROI 面积门回退后完整图已经误报的 `24` 个。
- 5 折分割器 Dice 更高但没有提升 demo 指标，原因更像是 ROI 面积分布和融合器校准不匹配；诊断分支不是单纯吃分割 Dice。
- 下一步如果要优化，只能在 BUSBRA OOF 内做恶性 FN 风格的增强、良性 hard negative、ROI 面积门和融合器重选；BUSI 错误样本不得回灌训练或参与配置选择。
