<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 基础模型公平性复核

> 本报告只用于重新整理已有模型测试结果。BUSBRA 用于训练和内部验证，BUSI 只用于外部评估；基础模型排序不混入 BUSI 阈值搜索、TTA 或最终 ensemble 优化结果。

日期：2026-04-25

## 目的

本报告按照更公平的比较规则重新组织已有模型测试结果。目标是理解各模型家族的基础性能，而不是把原始 fold-1 模型和最终优化 ensemble 直接比较。

## 公平比较规则

只有同时满足以下条件的结果，才被视为严格的 basic-model comparison：

- Same dataset role：仅使用 BUSBRA 进行训练和验证。
- Same split：fold 1，1500 张训练图像和 375 张验证图像。
- Same input recipe：224 image size、CLAHE、horizontal flip augmentation。
- Same training budget：20 epochs。
- Same decision threshold：Sensitivity、Specificity 和 Accuracy 均使用 threshold 0.50。
- Same model form：单个模型；不包含 five-fold ensemble、mixed ensemble、threshold tuning 或 TTA-only enhancement comparison。

在该规则下，最可靠的基础对比来源是 fold-1 model comparison run。

## 严格 Fold-1 基础对比

| Rank | Model | AUC | Sensitivity | Specificity | Accuracy | Basic Finding |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | EfficientNetV2-S | 0.8937 | 0.7049 | 0.8775 | 0.8213 | strict single-model baselines 中 AUC 最高 |
| 2 | DenseNet121 | 0.8867 | 0.7213 | 0.8775 | 0.8267 | 非常接近 EfficientNet，Sensitivity 和 Accuracy 略高 |
| 3 | ResNet18 | 0.8756 | 0.7131 | 0.8577 | 0.8107 | 稳定、轻量且较强的 baseline |
| 4 | MobileNetV3-Small | 0.8704 | 0.6557 | 0.9170 | 0.8320 | compact baselines 中 Accuracy/Specificity 最好，但 Sensitivity 较低 |
| 5 | Basic CNN | 0.6427 | 0.0000 | 0.9921 | 0.6693 | 过弱；threshold 0.50 下几乎没有 malignant recall |
| 6 | VGG16 | 0.5000 | 0.0000 | 1.0000 | 0.6747 | 本次运行未学到有效区分能力 |
| - | AlexNet | - | - | - | - | 记录运行时 fallback support 尚未加入，运行失败 |

## 结果解读

严格比较显示，可用 baseline group 实际上很清晰：EfficientNetV2-S、DenseNet121、ResNet18 和 MobileNetV3-Small 都处于 0.87 到 0.89 AUC 区间。EfficientNetV2-S 按 AUC 排名第一，但 DenseNet121 足够接近，可以作为合理的互补模型。这支持后续将 EfficientNetV2-S 作为 main backbone、DenseNet121 作为 ensemble partner 的决策。

MobileNetV3-Small 不是 AUC 最高的模型，但 Specificity 和 Accuracy 很高。如果项目后续需要 CPU-friendly demo 的轻量模型，MobileNetV3-Small 比 Basic CNN 或 VGG16 更值得考虑。

Basic CNN 和 VGG16 在当前 recipe 下不应被视为严肃的最终候选。它们的 Accuracy 有误导性，因为验证集中 benign 多于 malignant；默认阈值下两个模型几乎都漏掉了 malignant 样本。

## 扩展 Fold-1 参考结果

以下模型是在原始 comparison run 之后测试的。它们适合做 screening，但不属于严格原始对比表，因为它们是单独启动的实验；其中 Swin 使用了不同 learning rate。

| Model | AUC | Sensitivity | Specificity | Accuracy | Difference From Strict Rule | Finding |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Swin-Tiny | 0.7950 | 0.8852 | 0.5020 | 0.6267 | lr=0.0001，separate run | Sensitivity 高，但 false positives 很多；该 recipe 下不值得直接五折 |
| ConvNeXt-Tiny | 0.4519 | 0.0000 | 1.0000 | 0.6747 | separate run | threshold 0.50 下坍缩为全 benign prediction |

这些结果不意味着 Transformer 或 ConvNeXt 家族永远不可用。它们只说明当前快速 recipe 没有达到 EfficientNetV2-S / DenseNet121 / ResNet18 / MobileNetV3 组的竞争力。

重要限制：Swin-Tiny 和 ConvNeXt-Tiny 没有使用其模型家族推荐的完整 recipe。它们复用了项目中的简单分类器训练流水线，只用于快速初筛。当前 pipeline 没有应用每个 timm 模型的 pretrained normalization statistics、crop policy、scheduler、warmup、layer-wise learning-rate decay 或 model-specific augmentation recipe。因此这两个结果应理解为“quick project-recipe screening results”，不能视作 Swin 或 ConvNeXt 的性能上限。

## 不应混入基础 Backbone 排名的结果

以下结果本身有价值，但回答的是不同问题，不应混入基础单模型排名：

| Result Type | Example | Why It Is Not A Basic Backbone Comparison |
| --- | --- | --- |
| Longer EfficientNet training | EfficientNetV2-S fold1 AUC 0.9248 | 30-epoch final training，不是相同的 20-epoch comparison budget |
| Five-fold model evidence | EfficientNetV2-S five-fold mean AUC 0.8946 | 多折稳健性结果，不是单 fold architecture screening |
| Resolution experiments | 256 / 320 EfficientNetV2-S runs | 测试 input size 和 augmentation，不只测试 backbone choice |
| Sensitivity tuning | Sensitive / balanced-sensitive EfficientNet runs | 改变 class weighting 和 checkpoint selection objective |
| BUSI TTA / threshold results | TTA and threshold sweep reports | 测试 inference operating point，不是训练 backbone 本身 |
| Mixed ensembles | EfficientNetV2-S + DenseNet121 final ensemble | 多模型和阈值组合，性能更强但不是 basic model |

## 实用结论

对于基础模型性能，目前证据支持以下排序：

1. EfficientNetV2-S 是 fold-1 AUC 最好的 main backbone。
2. DenseNet121 是最接近的强互补模型。
3. ResNet18 和 MobileNetV3-Small 是强 secondary baselines。
4. Swin-Tiny 当前高 recall 但低 specificity；没有更好 recipe 前，不适合作为最终候选。
5. ConvNeXt-Tiny、Basic CNN 和 VGG16 在已测试 recipe 下不可用。

最公平的下一步模型实验不应是另一次 final-ensemble comparison，而应是新的严格 fold-1 single-model comparison：任何新候选都使用相同 split、image size、epoch budget、augmentation 和 threshold reporting。如果要认真判断 Swin 或 ConvNeXt，应进行第二层对比，加入 timm pretrained normalization、best-checkpoint selection、更低 learning rate、warmup/scheduler 和 class-balance control。
