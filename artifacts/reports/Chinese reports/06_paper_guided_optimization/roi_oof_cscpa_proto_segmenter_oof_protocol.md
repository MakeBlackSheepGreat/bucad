# ROI 裁剪与 OOF 融合实验报告

日期：2026-04-25

## 实验边界

- BUSBRA 使用真值 mask 生成 ROI，并通过 OOF 方式训练融合器。
- BUSI 的 `oracle` 结果使用 BUSI 真值 mask，只能视为 ROI 上限，不可作为可部署成绩。
- BUSI 的 `segmenter` 结果使用现有 `segmenter_fold1.pt` 预测 mask，是更接近部署口径的外部验证。
- 当前 `demo.yml` 和主线权重没有修改。

## OOF 融合器

- 特征：完整图两模型概率 + ROI 两模型概率。
- 特征模式：`logit`。
- OOF CV AUC：`0.9243`。
- OOF 推荐阈值：`0.54`。

## 外部验证结果

| 方案 | ROI来源 | ROI-only AUC | Stacking AUC | Stacking@OOF阈值 Sens | Stacking@OOF阈值 Spec | Stacking最佳阈值 | Stacking最佳 Sens | Stacking最佳 Spec | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| oracle | BUSI真值mask | 0.9037 | 0.9198 | 0.8333 | 0.8467 | 0.52 | 0.8762 | 0.8169 | 上限参考，不可部署 |
| segmenter | segmenter预测mask | 0.9154 | 0.9186 | 0.8286 | 0.8330 | 0.50 | 0.8905 | 0.8055 | 可部署代理，但依赖分割质量 |

## ROI 面积分布

| 方案 | fallback数量 | 平均面积占比 | 中位面积占比 | P10 | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 0 | 0.4333 | 0.3427 | 0.0511 | 0.9957 |
| segmenter | 0 | 0.7520 | 0.8973 | 0.2422 | 1.0000 |

## 结论

- 当前完整图主线 AUC：`0.9151`。
- segmenter ROI + OOF 融合 AUC：`0.9186`，高于完整图主线，说明 ROI/病灶区域引导方向有效。
- segmenter ROI 的中位面积占比偏大，说明当前不是“紧贴病灶”的精细裁剪，而是带较多上下文的病灶区域引导。
- 当前结果只能证明 ROI 方向值得继续验证，不建议立刻修改 Demo；下一步应在 BUSBRA OOF 内完成 ROI 参数、融合权重和运行阈值选择，再用 BUSI 做一次锁定配置后的外部评估。
