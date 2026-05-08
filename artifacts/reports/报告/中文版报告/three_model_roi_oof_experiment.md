# 三模型 ROI 与 OOF 融合旁路实验

日期：2026-04-26

## 实验目标

- 在不修改主线 `demo.yml` 的前提下，检查三模型测试线是否能从 ROI 裁剪和 OOF 融合中受益。
- 三模型使用 `EfficientNetV2-S + DenseNet121 + ConvNeXt-Tiny`，权重为 `0.358 / 0.244 / 0.398`。
- ROI OOF 使用 mask 阈值 `0.50`；部署评估使用 mask 阈值 `0.40` 和现有 `segmenter_fold1.pt`。

## OOF 融合器

- 特征：三模型完整图概率 + 三模型 ROI 概率。
- 特征模式：`logit`。
- OOF CV AUC：`0.9288`。
- OOF 推荐阈值：`0.56`。

## 结果对比

| 方案 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 两模型 ROI OOF LCC 主线 | 0.9208 | 0.550 | 0.8524 | 0.8169 | 0.8284 | 0.6911 | 0.7633 |
| 三模型完整图基线 | 0.9162 | 0.453 | 0.7476 | 0.9291 | 0.8702 | 0.8351 | 0.7889 |
| 三模型 ROI OOF Stacking@OOF阈值 | 0.9210 | 0.560 | 0.8048 | 0.8810 | 0.8563 | 0.7647 | 0.7842 |
| 三模型 ROI OOF Stacking@Youden | 0.9210 | 0.550 | 0.8238 | 0.8719 | 0.8563 | 0.7555 | 0.7882 |
| 三模型 ROI OOF LCC 正式配置复跑 | 0.9229 | 0.560 | 0.8143 | 0.8673 | 0.8501 | 0.7467 | 0.7790 |
| 三模型 ROI-only@Youden | 0.9202 | 0.450 | 0.7905 | 0.9153 | 0.8748 | 0.8177 | 0.8039 |
| 三模型 oracle ROI Stacking@Youden | 0.9199 | 0.520 | 0.8333 | 0.8696 | 0.8578 | 0.7543 | 0.7919 |

## ROI 面积分布

| ROI来源 | fallback数量 | 平均面积占比 | 中位面积占比 | P10 | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 0 | 0.4333 | 0.3427 | 0.0511 | 0.9957 |
| segmenter | 0 | 0.7046 | 0.7969 | 0.2256 | 1.0000 |

## 结论

- 三模型完整图基线：`AUC 0.9162, Sens 0.7476, Spec 0.9291, Acc 0.8702, Precision 0.8351, F1 0.7889`。
- 三模型 segmenter ROI + OOF Stacking AUC：`0.9210`，相对三模型完整图基线变化 `+0.0048`。
- 与当前两模型 ROI OOF LCC 主线 AUC `0.9208` 相比，三模型 ROI OOF 变化 `+0.0002`。
- 正式配置 `configs/inference/ensemble_effnet_densenet_convnext_roi_oof_lcc_mask04.yml` 通过 `eval_busi.py` 复跑后 AUC 为 `0.9229`，高于当前两模型主线 `+0.0021`。
- 该配置的 Precision 和 F1 相比两模型主线更高，但 Sensitivity 从 `0.8524` 降到 `0.8143`，不建议直接替换主线；更适合作为“高 AUC / 更均衡误报”的测试线候选。
- 如果后续继续做三模型方向，优先改进 DenseNet 分支的 ROI 表现或改为三模型多特征 stacking，而不是直接替换当前 demo。
