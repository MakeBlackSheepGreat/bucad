<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 训练 Recipe 审计

日期：2026-04-25

## 问题

之前训练过的模型，是否也存在 quick Swin-Tiny 和 ConvNeXt-Tiny 实验中发现的 training-recipe 问题？

## 简短结论

部分存在。

之前的 timm-based models 使用的是项目通用 classifier recipe，而不是每个模型家族完整推荐的 timm fine-tune recipe。这会影响我们判断各 backbone 绝对潜力的力度。但它并不会否定项目的数据边界、case-level split，也不会否定现有 EfficientNetV2-S 和 DenseNet121 在项目 recipe 下取得了有效结果。

## 共同 Recipe 局限

| Area | Current Project Behavior | Why It Matters |
| --- | --- | --- |
| Normalization | Images are scaled to `[0, 1]` only | pretrained timm backbones 通常期望 ImageNet 或 model-specific mean/std |
| Model-specific transform | One shared resize pipeline is used | timm models 有不同 interpolation/crop/input-size defaults |
| Scheduler/warmup | Fixed learning rate AdamW | Transformer/ConvNeXt-style models 往往需要 warmup 和 cosine decay |
| Checkpoint selection | Default is last checkpoint | 小型医学数据集中 best AUC / best Youden checkpoint 可能更优 |
| Hyperparameter reuse | Same simple recipe reused across families | 适合快速 screening，但不足以判断一个模型家族的性能上限 |

## 当前代码证据

- classifier transform 会 resize，按配置应用 CLAHE，除以 255，并转换为 CHW tensor。
- training loop 使用 AdamW 和固定 learning rate，默认 `checkpoint_strategy = last`。
- EfficientNetV2-S、DenseNet121、Swin-Tiny 和 ConvNeXt-Tiny 都使用项目级 training pipeline，而不是 model-specific timm recipe。

## 模型间差异

该问题对不同模型的严重程度并不相同。

| Model Family | Current Evidence | Risk Interpretation |
| --- | --- | --- |
| EfficientNetV2-S | 已有较强 fold 和 BUSI 结果 | Recipe 不是完全最优，但模型在本项目中明确有效 |
| DenseNet121 | fold 结果强，并且能作为有用 ensemble complement | 也未完全优化，但在当前 recipe 下有效 |
| ResNet18 / MobileNetV3 | fold-1 baselines 合理 | 可作为 project-recipe baselines |
| Swin-Tiny | Sensitivity 高但 Specificity 低 | 当前 recipe 可能不适合，不能判断 Swin family ceiling |
| ConvNeXt-Tiny | threshold 0.50 下坍缩为全 benign | 强烈提示当前 recipe 对该运行不合适 |
| Basic CNN / VGG16 | 当前 recipe 下较弱 | 没有大幅 recipe 改造前，不是严肃最终候选 |

## 仍然有效的内容

- BUSBRA/BUSI data boundary 仍然有效。
- BUSI 没有用于训练。
- Case-level fold split 仍然有效。
- fold-1 basic comparison 仍可作为“same project recipe”比较。
- final mixed ensemble result 仍然是当前 inference pipeline 下的有效经验结果。

## 需要降级表述的内容

- 不应声称 Swin-Tiny 或 ConvNeXt-Tiny 天然不适合。
- 不应声称当前 fold-1 basic comparison 代表各模型最佳可能性能。
- 不应把 quick model-family screening 结果直接和 final optimized ensemble 对比。
- 不应把 224-only 和 fixed-LR 结果视为那些推荐 recipe 差异很大的模型的最终证据。

## 推荐修复

增加第二层 “timm recommended recipe” 实验路径：

1. 使用 `timm.data.resolve_model_data_config()` 或等价方式提取 model-specific input size、interpolation、crop policy、mean 和 std。
2. 增加 configurable normalization：`none`、`imagenet`、`timm` 或 `custom`。
3. 增加 scheduler support：warmup + cosine decay。
4. 增加明确 best checkpoint selection：best AUC、best Youden 或 sensitivity-constrained score。
5. 先对 EfficientNetV2-S、DenseNet121、Swin-Tiny 和 ConvNeXt-Tiny 在 improved recipe 下只跑 fold1。
6. 只有 fold1 有竞争力时，再投入五折训练。

## 实用结论

早期结果不是没用，但必须准确标注：

- “Project recipe v1” results：可用于比较当前 implemented pipeline 下的模型表现。
- “Model ceiling” results：Swin/ConvNeXt 还没有；其他 timm backbones 也不算完全充分。

下一步严肃工作是实现 timm-aware training recipe，并对最强已有候选和新候选 backbones 重跑 fold1。
## 比赛指标完整性说明

比赛统一指标集为 AUC、Accuracy、Recall/Sensitivity、Precision、Specificity 和 F1-Score。如果历史归档表中 Precision 或 F1-Score 显示 `-`，说明原始摘要未保留可反推该值的 confusion matrix 或原始概率。锁定 BUSI 运行点结果以 `artifacts/reports/competition_metrics_all_busi_reports.md` 作为完整指标来源。
