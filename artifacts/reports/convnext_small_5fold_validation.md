# ConvNeXt-Small 强正则与五折验证实验报告

> 数据边界说明：BUSBRA 用于训练、内部验证、OOF、模型选择、阈值选择和 ROI 参数选择；BUSI 是锁定外部评估集，只用于最终外部验证，不用于训练或调参。


日期：2026-04-25

## 实验边界

- 本报告只记录主模型升级的旁路测试，不修改 `configs/inference/demo.yml`，不改变当前 Demo 主线。
- 训练仍严格只使用 BUSBRA 训练/内部验证划分；BUSI 只用于外部评估、阈值分析和误差观察。
- 对比时区分“单模型/五折/集成”三类结果，避免把未优化单模型直接和已经优化过的正式集成方案混为一谈。

## 本轮测试目标

1. 验证 ConvNeXt-Small 是否能作为比 ConvNeXt-Tiny 更强的主模型。
2. 验证更强正则是否能缓解 Small 参数量变大后的过拟合。
3. 扩展到五折后，观察 Small 的整体泛化能力是否能超过当前主线。
4. 离线检查 Small 是否能作为低权重辅助模型，为现有集成提供互补收益。

## 配置概览

| 配置 | 文件 | 说明 |
| --- | --- | --- |
| Small 基础训练 | `configs/classifier/convnext_small_timm_recipe.yml` | ConvNeXt-Small，沿用 timm 推荐预处理方向，作为升级主模型基线 |
| Small 强正则训练 | `configs/classifier/convnext_small_timm_recipe_regularized.yml` | 降低学习率，增加 weight decay、drop path、label smoothing 和轻量增强 |
| Small fold1 TTA | `configs/inference/convnext_small_timm_recipe_fold1_tta_crop_sweep.yml` | fold1 单模型 crop-sweep TTA |
| Small 五折 TTA | `configs/inference/convnext_small_timm_recipe_5fold_tta_crop_sweep.yml` | 五折平均 + crop-sweep TTA |

## 内部五折验证结果

| 模型 | Fold | AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy | 最佳轮次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Small | 1 | 0.9118 | 0.7295 | - | - | 0.9091 | 0.8507 | 28 |
| ConvNeXt-Small | 2 | 0.9177 | 0.8926 | - | - | 0.8150 | 0.8400 | 8 |
| ConvNeXt-Small | 3 | 0.9444 | 0.7603 | - | - | 0.8976 | 0.8533 | 10 |
| ConvNeXt-Small | 4 | 0.8936 | 0.8279 | - | - | 0.8458 | 0.8400 | 4 |
| ConvNeXt-Small | 5 | 0.9135 | 0.7934 | - | - | 0.8976 | 0.8640 | 16 |
| ConvNeXt-Small 均值 | 1-5 | 0.9162 | 0.8007 | - | - | 0.8730 | 0.8496 | - |

内部五折的均值并不差，但 fold4 明显偏低，且多个 fold 的最佳轮次较早，说明更大模型确实更容易在当前数据规模下提前进入过拟合区间。

## BUSI 外部评估结果

| 方案 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | 混淆矩阵 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Small fold1 crop-sweep | 0.8991 | 0.50 | 0.7857 | 0.8673 | 0.8408 | 0.7399 | 0.7621 | TN 379 / FP 58 / FN 45 / TP 165 |
| ConvNeXt-Small fold1 强正则 crop-sweep | 0.8759 | 0.50 | 0.7714 | 0.8650 | 0.8346 | 0.7330 | 0.7517 | TN 378 / FP 59 / FN 48 / TP 162 |
| ConvNeXt-Small 五折 crop-sweep | 0.9040 | 0.50 | 0.8190 | 0.8535 | 0.8423 | 0.7288 | 0.7713 | TN 373 / FP 64 / FN 38 / TP 172 |
| ConvNeXt-Tiny 五折 crop-sweep | 0.9054 | 0.50 | 0.7714 | 0.8879 | 0.8501 | 0.7678 | 0.7696 | TN 388 / FP 49 / FN 48 / TP 162 |
| 当前两模型正式方案 | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |
| 当前三模型优化方案 | 0.9162 | 0.453 | 0.7476 | 0.9291 | 0.8702 | 0.8351 | 0.7889 | TN 406 / FP 31 / FN 53 / TP 157 |

## 阈值观察

Small 五折在 BUSI 上的 Youden 最优阈值为 `0.47`：

| 阈值策略 | 阈值 | Sensitivity | Specificity | Accuracy | Youden J |
| --- | ---: | ---: | ---: | ---: | ---: |
| 默认阈值 | 0.50 | 0.8190 | 0.8535 | 0.8423 | 0.6726 |
| Youden 最优 | 0.47 | 0.8333 | 0.8467 | 0.8423 | 0.6800 |
| 偏准确率点 | 0.58 | 0.7810 | 0.8879 | 0.8532 | 0.6688 |

Small 五折的主要特点是敏感性较好，但特异性不如当前主线。它能减少漏诊，但会带来更多假阳性。

## 离线概率融合观察

基于已有 BUSI 输出概率做离线加权搜索，仅用于判断互补性，不作为正式训练或最终结论。

| 融合方式 | Small 权重 | AUC | 推荐阈值 | Sensitivity | Precision | F1-Score | Specificity | Accuracy | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 两模型正式方案 + Small | 0.17 | 0.9153 | 0.401 | 0.8095 | - | - | 0.8719 | 0.8516 | AUC 只提升约 0.0002，收益很弱 |
| 三模型优化方案 + Small | 0.20 | 0.9165 | 0.343 | 0.8333 | - | - | 0.8535 | 0.8470 | AUC 只提升约 0.0003，但明显牺牲特异性 |
| Tiny 五折 + Small | 0.43 | 0.9085 | 0.391 | 0.8524 | - | - | 0.8330 | 0.8393 | 更偏向召回，整体不如正式集成 |

Small 存在一定互补性，但互补价值主要体现在提高 Sensitivity，而不是稳定提高 AUC。当前证据不足以支持把 Small 提升为正式主模型。

## 结论

- 强正则版本没有带来收益：fold1 外部 AUC 从基础 Small 的 `0.8991` 降到 `0.8759`，说明当前正则强度或训练策略并不适合直接扩大。
- Small 五折比 Small fold1 更稳，外部 AUC 提升到 `0.9040`，但仍低于 ConvNeXt-Tiny 五折 `0.9054` 和当前正式两模型方案 `0.9151`。
- Small 的优势是 Sensitivity，默认阈值下达到 `0.8190`，但代价是 Specificity 降到 `0.8535`。
- 当前主线不建议替换。ConvNeXt-Tiny + EfficientNetV2-S 仍是更稳的 Demo 主方案。

## 下一步建议

1. 不继续盲目加大 ConvNeXt-Small 正则；它已经出现“内部可看、外部下降”的信号。
2. 如果目标是提高 AUC，优先做当前两模型主线的 seed ensemble、OOF 校准或轻量 stacking，而不是继续扩大单模型。
3. 如果目标是提高 Sensitivity，可以保留 Small 作为低权重辅助分支或单独提供“高召回模式”，但必须明确会牺牲特异性。
4. 后续更值得尝试的是基于 BUSBRA OOF 的阈值/温度校准、hard case 复盘、ROI/分割辅助裁剪，而不是继续简单堆模型容量。
