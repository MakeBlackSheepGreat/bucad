<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->
# Model-Zoo OOF 候选方案协议

## 数据边界

- 筛选仅使用 BUSBRA OOF 预测。
- 本脚本不读取 BUSI 数据，仅在该候选方案冻结后方可使用。
- ROI 分支保持可部署的 EfficientNetV2-S + ConvNeXt-Tiny ROI 组合；新的 model-zoo 成员仅影响全图分支。

## 视图

| 视图 | 角色 |
| --- | --- |
| eff_identity | 全图 model-zoo 候选 |
| conv_crop_sweep | 全图 model-zoo 候选 |
| densenet_identity | 全图 model-zoo 候选 |
| convsmall_crop_sweep | 全图 model-zoo 候选 |
| swin_crop_sweep | 全图 model-zoo 候选 |

## OOF 指标

| 方案 | AUC | 阈值 | Sensitivity | Specificity | 准确率 | Precision | F1 | 混淆矩阵 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 当前 demo OOF | 0.9166 | 0.510 | 0.8402 | 0.8423 | 0.8416 | 0.7183 | 0.7745 | TN 1068 / FP 200 / FN 97 / TP 510 |
| 选定嵌套 OOF | 0.9352 | 0.410 | 0.8402 | 0.8927 | 0.8757 | 0.7895 | 0.8140 | TN 1132 / FP 136 / FN 97 / TP 510 |
| 选定 refit OOF | 0.9369 | 0.410 | 0.8369 | 0.8967 | 0.8773 | 0.7950 | 0.8154 | TN 1137 / FP 131 / FN 99 / TP 508 |

## 选定候选方案

- 名称：`w_0.31_0.03_0.23_0.04_0.39_gate0.08-0.90_C1_none`
- 候选配置文件：`configs/inference/demo_model_zoo_oof_candidate.yml`
- Area gate：min `0.08`，max `0.90`，回退 `206` 个 OOF 样本。
- Logistic C：`1.0`，class_weight：`None`

| 全图视图 | 权重 |
| --- | ---: |
| eff_identity | 0.3057 |
| conv_crop_sweep | 0.0321 |
| densenet_identity | 0.2315 |
| convsmall_crop_sweep | 0.0400 |
| swin_crop_sweep | 0.3907 |

## 候选方案排名

| 排名 | AUC | 阈值 | Sens | Spec | Precision | F1 | Gate | 权重 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.9352 | 0.410 | 0.8402 | 0.8927 | 0.7895 | 0.8140 | 0.08-0.90 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 2 | 0.9352 | 0.570 | 0.8451 | 0.8904 | 0.7868 | 0.8149 | 0.08-1.01 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 3 | 0.9351 | 0.390 | 0.8468 | 0.8841 | 0.7776 | 0.8107 | 0.08-0.90 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 4 | 0.9351 | 0.380 | 0.8402 | 0.8849 | 0.7774 | 0.8076 | 0.08-0.90 | eff_identity=0.15, conv_crop_sweep=0.14, densenet_identity=0.24, convsmall_crop_sweep=0.01, swin_crop_sweep=0.46 |
| 5 | 0.9351 | 0.390 | 0.8501 | 0.8825 | 0.7759 | 0.8113 | 0.08-1.01 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 6 | 0.9351 | 0.560 | 0.8484 | 0.8849 | 0.7791 | 0.8123 | 0.08-1.01 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 7 | 0.9351 | 0.380 | 0.8418 | 0.8856 | 0.7790 | 0.8092 | 0.08-0.90 | eff_identity=0.15, conv_crop_sweep=0.14, densenet_identity=0.24, convsmall_crop_sweep=0.01, swin_crop_sweep=0.46 |
| 8 | 0.9351 | 0.380 | 0.8550 | 0.8809 | 0.7746 | 0.8128 | 0.10-1.01 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |
| 9 | 0.9350 | 0.550 | 0.8435 | 0.8785 | 0.7688 | 0.8044 | 0.08-1.01 | eff_identity=0.25, conv_crop_sweep=0.03, densenet_identity=0.33, convsmall_crop_sweep=0.10, swin_crop_sweep=0.30 |
| 10 | 0.9350 | 0.410 | 0.8402 | 0.8880 | 0.7822 | 0.8102 | 0.08-1.01 | eff_identity=0.31, conv_crop_sweep=0.03, densenet_identity=0.23, convsmall_crop_sweep=0.04, swin_crop_sweep=0.39 |

## 决策规则

- 如果冻结的 BUSI 外部检查在 AUC 上有所提升，且未明显损害 Sensitivity、Precision、Specificity 或 F1，则可考虑将该候选方案合并至 demo。
- 如果 BUSI 未有改善，则将其保留为内部实验，且不使用 BUSI 反馈调整候选方案。
