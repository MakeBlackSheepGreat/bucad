# 论文启发分割与 ROI 优化实验总结

## 1. 实验边界

- 主线配置 `configs/inference/demo.yml` 未修改。
- BUSBRA 只用于训练、内部验证、OOF、ROI 融合器拟合和候选筛选。
- BUSI 只用于冻结配置后的外部复核；外验结果没有回流到阈值、ROI gate、stacker 或模型参数。
- 当前主线 BUSI 对照：AUC `0.9256`，threshold `0.51`，Sensitivity `0.8667`，Specificity `0.8467`，F1 `0.7930`。

## 2. 已落地的工程能力

- 新增 CENet-lite 分割模块：DSEB 边缘增强跳连、差分注意力、CFAM 通道校准、多尺度上下文聚合、轻量 non-local。
- 新增 PVT-v2 CENet 尝试入口：基于 `timm` `features_only` 的特征金字塔编码器。
- 新增 CSC-PA labeled-only 近似：PAL-lite 像素邻域亲和损失、前景原型对比损失、边界原型对比损失。
- 新增分割指标：Dice、IoU、Boundary F1、HD95。
- 新增论文实验编排脚本：`scripts/run_paper_guided_segmentation_protocol.py`。
- 扩展 ROI OOF 协议：`scripts/run_roi_oof_experiment.py` 支持 `--roi-mask-source segmenter_oof`，可以用候选分割器的 OOF 预测 mask 重训 ROI stacker。

## 3. Stage 1 全方法 screening

协议：每个方法 `1` fold、`1` epoch，冻结后各跑一次 BUSI 外部复核。该阶段用于快速筛掉明显外迁不足的方法，不作为最终训练结论。

| 方法 | 内部 Dice | Boundary F1 | BUSI AUC | Sensitivity | Specificity | F1 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_unet_bce | 0.7906 | 0.5131 | 0.9197 | 0.8476 | 0.8535 | 0.7876 | 不合入 |
| baseline_unet_bce_dice | 0.8309 | 0.5983 | 0.9195 | 0.8619 | 0.8215 | 0.7719 | 不合入 |
| cenet_dseb_only | 0.6190 | 0.2917 | 0.9159 | 0.7571 | 0.8833 | 0.7571 | 不合入 |
| cenet_cfam_only | 0.7006 | 0.3196 | 0.9046 | 0.7905 | 0.8741 | 0.7703 | 不合入 |
| cenet_dseb_cfam | 0.7151 | 0.4210 | 0.9118 | 0.7952 | 0.8764 | 0.7749 | 不合入 |
| cenet_boundary_head | 0.7228 | 0.4863 | 0.9116 | 0.8429 | 0.8284 | 0.7662 | 不合入 |
| cscpa_pal_lite | 0.7784 | 0.5106 | 0.9182 | 0.8571 | 0.8352 | 0.7792 | 不合入 |
| cscpa_foreground_edge_proto | 0.7799 | 0.4803 | 0.9239 | 0.8714 | 0.8421 | 0.7922 | 最接近主线，进入正式复核 |
| cenet_pvtv2_boundary_pal | 0.5421 | 0.3963 | 0.9108 | 0.8714 | 0.8146 | 0.7722 | 不合入 |

Stage 1 最高 AUC 为 `cscpa_foreground_edge_proto` 的 `0.9239`，仍低于主线 `0.9256`，但 Sensitivity 达到 `0.8714`，因此进入 Stage 2。

## 4. Stage 2 正式候选复核

方法：`cscpa_foreground_edge_proto`。

训练协议：BUSBRA `5` fold、每 fold `5` epoch。外部复核：冻结 5 个分割 checkpoint 后跑一次 BUSI。

内部 5 折分割指标：

| Dice | IoU | Boundary F1 | HD95 |
| ---: | ---: | ---: | ---: |
| 0.8693 | 0.7856 | 0.6796 | 15.44 |

BUSI 外部结果：

| AUC | Threshold | Sensitivity | Specificity | Precision | F1 | Confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.9165 | 0.51 | 0.8381 | 0.8467 | 0.7243 | 0.7770 | TN 370 / FP 67 / FN 34 / TP 176 |

结论：虽然内部 Dice 达到 `0.8693`，但 BUSI AUC 从主线 `0.9256` 降到 `0.9165`，Sensitivity 和 F1 也下降，因此不进入主线候选。

## 5. 严格 ROI OOF 重训复核

为避免“只换分割器但沿用旧 stacker/gate”的不公平问题，对 `cscpa_foreground_edge_proto` 追加了预测 mask OOF 协议：

- BUSBRA 内部 ROI：使用候选 5 折分割器对各折验证样本预测 mask，不使用 GT mask。
- ROI 分类概率：重新生成 ROI OOF cache。
- ROI 融合器：基于 BUSBRA OOF 重新拟合 logit stacker，OOF CV AUC `0.9243`，OOF 阈值 `0.54`。
- BUSI 外部：冻结后仅复核一次。

严格 ROI OOF 结果：

| 口径 | BUSI AUC | Threshold | Sensitivity | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| segmenter ROI + refit stacker @ OOF threshold | 0.9186 | 0.54 | 0.8286 | 0.8330 | 0.7615 |
| segmenter ROI + refit stacker @ best Youden | 0.9186 | 0.50 | 0.8905 | 0.8055 | 0.7759 |
| oracle ROI upper-bound reference | 0.9198 | - | - | - | - |

结论：重新拟合 ROI stacker 后仍未超过主线，且 oracle ROI 上限也只有 `0.9198`。这说明当前候选分割器的 ROI 分布并没有给分类主线带来可迁移收益。

## 6. 总体结论

- 本轮没有产生可合入主线 `demo.yml` 的候选。
- 最接近的方向是 CSC-PA 的前景/边界原型思想，但正式 5 折后外部 AUC 降到 `0.9165`，严格 ROI OOF 重训后也只有 `0.9186`。
- CENet/CSC-PA 思想在分割指标上有价值，尤其是边界与原型约束能提升 mask 质量；但当前比赛主指标由分类与 ROI 分布共同决定，单纯提高 Dice 不能保证诊断 AUC 提升。
- 后续如果继续优化，应优先研究“分割 mask 分布 -> ROI 裁剪 -> 分类概率校准”的耦合，而不是继续堆叠更复杂的分割模块。
