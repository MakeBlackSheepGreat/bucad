# ROI 裁剪与 OOF 融合实验报告

日期：2026-04-25

## 实验边界

- BUSBRA 使用真值 mask 生成 ROI，并通过 OOF 方式训练融合器。
- BUSI 的 `oracle` 结果使用 BUSI 真值 mask，只能视为 ROI 上限，不可作为可部署成绩。
- BUSI 的 `segmenter` 结果使用现有 `segmenter_fold1.pt` 预测 mask，是更接近部署口径的外部验证。
- 后续 `roi_oof_lcc_optimization.md` 在不使用 BUSI 调参的前提下，基于 BUSBRA 分割验证加入 `mask_threshold=0.40` 和最大连通域 ROI 后处理，并已合并到 `demo.yml`。

## OOF 融合器

- 特征：完整图两模型概率 + ROI 两模型概率。
- 特征模式：`logit`。
- OOF CV AUC：`0.9223`。
- OOF 推荐阈值：`0.55`。

## 外部验证结果

| 方案 | ROI来源 | ROI-only AUC | Stacking AUC | Stacking@OOF阈值 Sens | Stacking@OOF阈值 Spec | Stacking最佳阈值 | Stacking最佳 Sens | Stacking最佳 Spec | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| oracle | BUSI真值mask | 0.9037 | 0.9202 | 0.8286 | 0.8330 | 0.52 | 0.8905 | 0.8032 | 上限参考，不可部署 |
| segmenter | segmenter预测mask | 0.9196 | 0.9196 | 0.8429 | 0.8330 | 0.65 | 0.7810 | 0.9153 | 旧版可部署代理，未使用 LCC 后处理 |
| segmenter + LCC | segmenter预测mask + 最大连通域 | - | 0.9208 | 0.8524 | 0.8169 | 0.51 | - | - | 当前 `demo.yml` 主线，mask 阈值和 LCC 来自 BUSBRA 分割验证 |

## ROI 面积分布

| 方案 | fallback数量 | 平均面积占比 | 中位面积占比 | P10 | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 0 | 0.4333 | 0.3427 | 0.0511 | 0.9957 |
| segmenter | 3 | 0.6274 | 0.6936 | 0.1575 | 1.0000 |

## 结论

- 当前完整图主线 AUC：`0.9151`。
- segmenter ROI + OOF 融合 AUC：`0.9196`，高于完整图主线，说明 ROI/病灶区域引导方向有效。
- 加入 BUSBRA 分割验证选择出的 `mask_threshold=0.40` 和最大连通域后处理后，当前主线 AUC 提升到 `0.9208`，Sensitivity 提升到 `0.8524`。
- segmenter ROI 的中位面积占比偏大，说明当前不是“紧贴病灶”的精细裁剪，而是带较多上下文的病灶区域引导。
- 当前 ROI OOF 配置已按 BUSBRA OOF 固定参数和阈值，并完成 BUSI 锁定外部复核；由于 LCC ROI 后处理后的 AUC 和 Sensitivity 均高于上一版 ROI 主线，已合并为当前 demo 主线。
## Competition Metric Completeness Note

The official competition metric set is AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score. If an old archived table shows `-` for Precision or F1-Score, the historical summary did not preserve the confusion matrix or raw probabilities needed to reconstruct that value. For locked BUSI operating-point results, use `artifacts/reports/competition_metrics_all_busi_reports.md` as the complete metric source.
