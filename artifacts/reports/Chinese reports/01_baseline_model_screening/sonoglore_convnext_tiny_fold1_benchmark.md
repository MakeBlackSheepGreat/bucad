# SonoGloReNet-T fold1 基准实验报告

> 数据边界说明：BUSBRA 仅用于训练、内部验证和模型选择；BUSI 仅用于锁定外部评估。本报告采用单折 `fold1` 对照，比较对象为现有 `ConvNeXt-Tiny timm recipe fold1`。

日期：2026-06-18

## 实验目的

- 验证 `SonoGloReNet-T` 是否能够在当前项目主分类支线中稳定训练。
- 对比其与 `ConvNeXt-Tiny timm recipe fold1` 的内部验证表现和 BUSI 外部泛化表现。
- 观察新结构在推理耗时上的额外成本是否明显。

## 模型设置

### SonoGloReNet-T

- 配置：`configs/classifier/sonoglore_convnext_tiny.yml`
- 推理配置：`configs/inference/sonoglore_convnext_tiny_fold1.yml`
- 结构：
  - `ConvNeXt-Tiny features_only`
  - `stage4` 单层轻量自注意力
  - `stage2/3/4` 三尺度 `1x1 Conv + GELU + GeM`
  - `concat + LayerNorm + MLP` 二分类头

### 对照基线

- 训练报告：`artifacts/reports/train_cls_convnext_tiny_timm_recipe_fold1.json`
- 外部评估报告：`artifacts/reports/busi_convnext_tiny_timm_recipe_fold1.json`

## 训练结果对比

| 模型 | fold | 内部验证 AUC | Sensitivity | Specificity | Accuracy | Precision | F1-Score | 最优 epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny timm recipe | 1 | 0.9259 | 0.7213 | 0.9328 | 0.8640 | - | - | 13 |
| SonoGloReNet-T | 1 | 0.9096 | 0.7295 | 0.9012 | 0.8453 | 0.7807 | 0.7542 | 22 |

## BUSI 外部评估对比

默认阈值 `0.50`：

| 模型 | AUC | Sensitivity | Specificity | Accuracy | Precision | F1-Score | 混淆矩阵 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny timm recipe fold1 | 0.8926 | 0.7762 | 0.8696 | 0.8393 | 0.7409 | 0.7581 | TN 380 / FP 57 / FN 47 / TP 163 |
| SonoGloReNet-T fold1 | 0.8782 | 0.8048 | 0.7963 | 0.7991 | 0.6550 | 0.7222 | TN 348 / FP 89 / FN 41 / TP 169 |

Youden 最优阈值：

| 模型 | AUC | 推荐阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Youden J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny timm recipe fold1 | 0.8926 | 0.54 | 0.7667 | 0.8879 | 0.8485 | 0.7667 | 0.7667 | 0.6545 |
| SonoGloReNet-T fold1 | 0.8782 | 0.72 | 0.7810 | 0.8421 | 0.8223 | 0.7039 | 0.7404 | 0.6231 |

## 推理耗时

测试条件：

- 环境：`BUCAD`
- 设备：沿运行时自动选择
- 输入：同一张 BUSI 良恶性样本图像
- 设置：关闭分割和 Grad-CAM，仅测分类诊断
- 统计方式：预热 1 次后重复 5 次

| 模型 | 平均耗时 (s) | 最小耗时 (s) | 最大耗时 (s) |
| --- | ---: | ---: | ---: |
| ConvNeXt-Tiny timm recipe fold1 | 0.01785 | 0.01591 | 0.02216 |
| SonoGloReNet-T fold1 | 0.01771 | 0.01712 | 0.01810 |

观察：本轮 `SonoGloReNet-T` 在单张分类推理耗时上没有出现明显额外开销，和基线基本处于同一量级。

## 结果分析

- `SonoGloReNet-T` 可以稳定训练并完成 BUSI 外部评估，说明结构接入和运行链路是正确的。
- 但从当前 `fold1` 结果看，新增轻量注意力和多尺度头没有带来外部 AUC 提升。
- 与 `ConvNeXt-Tiny timm recipe fold1` 相比：
  - 内部验证 AUC 下降约 `0.0164`
  - BUSI AUC 下降约 `0.0144`
  - 外部 Sensitivity 略升，从 `0.7762` 到 `0.8048`
  - 外部 Specificity 明显下降，从 `0.8696` 到 `0.7963`
- 这说明当前版本更偏向召回恶性样本，但误报增多，整体排序能力和校准表现暂时弱于基线。

## 当前结论

- 以 `fold1` 为准，`SonoGloReNet-T v1` 尚未超过现有 `ConvNeXt-Tiny timm recipe` 基线。
- 该结构方向不是无效尝试，因为它几乎不增加推理成本，并且提升了外部召回率。
- 当前更合理的判断是：这是一条可继续优化的结构分支，还不适合直接替换主线骨干。

## 建议的下一步

1. 先完成 `SonoGloReNet-T 5-fold` 训练，确认当前结果是否只是单折波动。
2. 优先尝试更强正则化版本：
   - `drop_path_rate`
   - `label_smoothing`
   - `MixUp / CutMix`
   - 更保守学习率
3. 单独检查 `stage4 attention` 对外部阈值漂移的影响，必要时做消融：
   - 去掉 attention，仅保留多尺度池化
   - 仅保留 `stage3 + stage4`
4. 若 `5-fold` 仍落后，则将该结构保留为研究支线，不进入当前主线集成。
