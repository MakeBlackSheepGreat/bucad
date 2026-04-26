# 两模型 OOF Stacking 实验报告

> ???????????? BUSI ??????????????????? BUSI ???????TTA?????????????????????????????????? BUSBRA OOF/????????? BUSI ????????


日期：2026-04-25

## 实验边界

- OOF 融合器只使用 BUSBRA 五折验证样本的 out-of-fold 概率训练。
- BUSI 只用于外部评估，不参与融合器训练、交叉验证或阈值选择。
- 本报告是旁路优化实验，不修改 `configs/inference/demo.yml`。

## 输入视图

- `eff_identity`：EfficientNetV2-S 五折，identity TTA。
- `conv_crop_sweep`：ConvNeXt-Tiny 五折，crop-sweep TTA，对齐当前 Demo 主线。
- `conv_hflip`：ConvNeXt-Tiny 五折，hflip TTA，对齐上一轮 AUC 候选。

## OOF 融合结果

| 方案 | 特征 | CV AUC | OOF阈值 | BUSI AUC | BUSI@OOF阈值 Sens | BUSI@OOF阈值 Spec | BUSI@OOF阈值 Acc | BUSI最佳Youden阈值 | BUSI最佳Youden Sens | BUSI最佳Youden Spec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_crop_sweep | eff_identity + conv_crop_sweep | 0.9245 | 0.31 | 0.9130 | 0.9429 | 0.5629 | 0.6862 | 0.45 | 0.8619 | 0.8192 |
| auc_hflip | eff_identity + conv_hflip | 0.9249 | 0.33 | 0.9134 | 0.9333 | 0.6041 | 0.7110 | 0.45 | 0.7810 | 0.8947 |
| multi_view | eff_identity + conv_crop_sweep + conv_hflip | 0.9249 | 0.46 | 0.9138 | 0.9429 | 0.5744 | 0.6940 | 0.66 | 0.7952 | 0.8810 |

## 外部混淆矩阵

| 方案 | BUSI@OOF阈值 | BUSI最佳Youden点 |
| --- | --- | --- |
| current_crop_sweep | TN 246 / FP 191 / FN 12 / TP 198 | TN 358 / FP 79 / FN 29 / TP 181 |
| auc_hflip | TN 264 / FP 173 / FN 14 / TP 196 | TN 391 / FP 46 / FN 46 / TP 164 |
| multi_view | TN 251 / FP 186 / FN 12 / TP 198 | TN 385 / FP 52 / FN 43 / TP 167 |

## 结论

- 本轮 BUSI AUC 最高的是 `multi_view`，AUC `0.9138`。
- OOF Stacking 的训练流程比直接在 BUSI 上搜权重更规范，但是否替换主线仍要看 BUSI 外部指标是否真正提升。
- 如果 OOF 阈值迁移到 BUSI 后敏感性下降明显，应保留当前 Demo 主线，把该结果作为实验记录。
