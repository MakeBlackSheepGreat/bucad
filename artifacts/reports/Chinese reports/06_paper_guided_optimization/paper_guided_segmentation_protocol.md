# 论文启发分割与 ROI 优化实验记录

## 实验边界

- BUSBRA 用于训练、内部验证、OOF 和候选筛选。
- BUSI 只用于冻结配置后的单次外部复核；不得根据 BUSI 结果回调参数。
- 主线 `configs/inference/demo.yml` 不在本脚本中修改。

## 主线对照

- BUSI AUC：`0.9256`
- 阈值：`0.51`
- Sensitivity / Specificity / F1：`0.8667` / `0.8467` / `0.7930`

## 方法结果

| 方法 | 来源 | 状态 | 内部Dice | 内部IoU | Boundary F1 | HD95 | BUSI AUC | 决策 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cenet_dseb_only | CENet DSEB | trained | 0.5482 | 0.4067 | 0.2590 | 98.27 | - | report_only |

## 说明

- `cenet_pvtv2_boundary_pal` 依赖 timm 对 PVT-v2 `features_only` 的支持；若构建失败，报告会记录为依赖受限。
- `cscpa_foreground_edge_proto` 是 labeled-only 近似，不等同于 CSC-PA 原文的半监督跨图像原型注意力完整复现。
- 决策为 `candidate_hold_for_user` 的配置只表示可供人工复核，不会自动合入主线。
