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
| baseline_unet_bce | control | trained | 0.7906 | 0.6883 | 0.5131 | 23.80 | 0.9197 | report_only |
| baseline_unet_bce_dice | segmentation loss ablation | trained | 0.8309 | 0.7359 | 0.5983 | 18.49 | 0.9195 | report_only |
| cenet_dseb_only | CENet DSEB | trained | 0.6190 | 0.4778 | 0.2917 | 73.11 | 0.9159 | report_only |
| cenet_cfam_only | CENet CFAM | trained | 0.7006 | 0.5806 | 0.3196 | 44.01 | 0.9046 | report_only |
| cenet_dseb_cfam | CENet DSEB + CFAM | trained | 0.7151 | 0.5991 | 0.4210 | 40.28 | 0.9118 | report_only |
| cenet_boundary_head | CENet boundary enhancement | trained | 0.7228 | 0.6098 | 0.4863 | 35.76 | 0.9116 | report_only |
| cscpa_pal_lite | CSC-PA PAL | trained | 0.7784 | 0.6751 | 0.5106 | 28.74 | 0.9182 | report_only |
| cscpa_foreground_edge_proto | CSC-PA FPA + EPA | trained | 0.7799 | 0.6728 | 0.4803 | 27.23 | 0.9239 | report_only |
| cenet_pvtv2_boundary_pal | CENet PvT-v2 encoder | trained | 0.5421 | 0.4103 | 0.3963 | 49.86 | 0.9108 | report_only |

## 说明

- `cenet_pvtv2_boundary_pal` 依赖 timm 对 PVT-v2 `features_only` 的支持；若构建失败，报告会记录为依赖受限。
- `cscpa_foreground_edge_proto` 是 labeled-only 近似，不等同于 CSC-PA 原文的半监督跨图像原型注意力完整复现。
- 决策为 `candidate_hold_for_user` 的配置只表示可供人工复核，不会自动合入主线。
