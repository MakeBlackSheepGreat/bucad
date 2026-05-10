<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->
# ROI 分割器 5-Fold 对比

日期：2026-04-26 23:29:10

## 决策

- 保持 `configs/inference/demo.yml` 使用当前的 `segmenter_fold1.pt` 运行时分割器。
- 不将 `segmenter_5fold_fold5.pt` 或 5-fold 掩膜集成合并至默认 demo。
- 5-fold 分割运行提升了验证 Dice，但生成的 ROI 掩膜在当前 ROI Area Gate 堆叠器下未能改善 BUSI 诊断指标。

## 固定阈值 BUSI 结果

| 候选方案 | AUC | 阈值 | 准确率 | Sensitivity | Precision | Specificity | F1-Score | 混淆矩阵 | Delta AUC | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| current_fold1_runtime_recheck | 0.9256 | 0.51 | 0.8532 | 0.8667 | 0.7309 | 0.8467 | 0.7930 | TN 370 / FP 67 / FN 28 / TP 182 | +0.0000 | +0.0000 |
| seg5fold_best_fold5 | 0.9181 | 0.51 | 0.8439 | 0.8524 | 0.7189 | 0.8398 | 0.7800 | TN 367 / FP 70 / FN 31 / TP 179 | -0.0075 | -0.0131 |
| seg5fold_mask_ensemble | 0.9200 | 0.51 | 0.8346 | 0.8524 | 0.7020 | 0.8261 | 0.7699 | TN 361 / FP 76 / FN 31 / TP 179 | -0.0055 | -0.0231 |

## 候选方案设置

- `current_fold1_runtime_recheck`：当前 `configs/inference/demo.yml`，在添加 segmenter-list 运行时支持后重新运行。
- `seg5fold_best_fold5`：`configs/inference/demo_roi_area_gate_seg5fold_best.yml`，在 BUSI 评估前通过 BUSBRA 分割 Dice 选取。
- `seg5fold_mask_ensemble`：`configs/inference/demo_roi_area_gate_seg5fold_ensemble.yml`，对 `segmenter_5fold_fold1.pt` 至 `segmenter_5fold_fold5.pt` 的 sigmoid 掩膜取平均。

## 解读

- Fold5 提升了分割验证 Dice，但其 BUSI ROI 诊断 AUC 较当前主线下降了 `0.0075`，F1 下降了 `0.0131`。
- 5-fold 掩膜集成同样表现不及当前主线，AUC 下降 `0.0055`，F1 下降 `0.0231`。
- 这表明当前 ROI 堆叠器和 area gate 已针对现有分割器掩膜分布进行了调优；仅凭更好的 Dice 不足以改善诊断 ROI 融合。
- 未来的改进方向应是在新的分割器掩膜上于 BUSBRA OOF 上重新训练或重新校准 ROI 堆叠器和 area gate，然后再进行锁定的 BUSI 检查。
