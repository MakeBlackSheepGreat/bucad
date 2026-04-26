<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# ConvNeXt-Tiny 实验

日期：2026-04-25

## 目标

评估 `convnext_tiny` 是否值得加入乳腺超声分类器候选池。本实验是 fold-1 单模型实验，只用于 backbone 初筛，不应直接拿来和最终优化后的 mixed ensemble 对比。

## 实验设置

| 项目 | 设置 |
| --- | --- |
| Model | `convnext_tiny` |
| Training data | BUSBRA only |
| Fold | 1 |
| Epochs | 20 |
| Input size | 224 |
| Preprocessing | CLAHE |
| Augmentation | Horizontal flip |
| Learning rate | 0.0003 |
| BUSI role | 仅作外部评估，不参与训练 |

## Recipe 局限

这次实验是快速 project-recipe 初筛，不是完整的 ConvNeXt 专项优化。实验复用了当前分类器流水线：square resize、CLAHE、horizontal flip、AdamW、固定 learning rate、无 scheduler/warmup，并保存 last checkpoint。

该配置没有完整使用 timm 针对 ConvNeXt 的 pretrained transform 和训练 recipe，例如 ImageNet mean/std normalization、crop policy、更低的 fine-tuning learning rate、warmup/scheduler、layer-wise learning-rate decay 等。

因此，本结果只能说明当前快速 ConvNeXt recipe 失败，不能证明 ConvNeXt 本身不适合该任务。

## Fold-1 单模型对比

| Model | AUC | Sensitivity | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| EfficientNetV2-S | 0.8937 | 0.7049 | 0.8775 | 0.8213 |
| DenseNet121 | 0.8867 | 0.7213 | 0.8775 | 0.8267 |
| ResNet18 | 0.8756 | 0.7131 | 0.8577 | 0.8107 |
| MobileNetV3-Small | 0.8704 | 0.6557 | 0.9170 | 0.8320 |
| Swin-Tiny | 0.7950 | 0.8852 | 0.5020 | 0.6267 |
| Basic CNN | 0.6427 | 0.0000 | 0.9921 | 0.6693 |
| VGG16 | 0.5000 | 0.0000 | 1.0000 | 0.6747 |
| ConvNeXt-Tiny | 0.4519 | 0.0000 | 1.0000 | 0.6747 |

在当前训练 recipe 下，ConvNeXt-Tiny 没有学到有效的 malignant 判别边界。最终 checkpoint 在 threshold 0.50 下把所有 validation 样本都判为 benign。实验中观测到的最佳 epoch 是 epoch 3，但其 AUC 也只有 0.5455，并且默认阈值下仍然没有召回 malignant 样本。

## BUSI 外部 Sanity Check

| Operating Point | AUC | Threshold | Sensitivity | Specificity | Accuracy | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default threshold | 0.6012 | 0.50 | 0.0000 | 1.0000 | 0.6754 | TN 437 / FP 0 / FN 210 / TP 0 |
| Best sweep point | 0.6012 | 0.10 | 1.0000 | 0.0000 | 0.3246 | TN 0 / FP 437 / FN 0 / TP 210 |

BUSI 概率在常规阈值下校准很差。threshold 0.50 时模型把全部样本判为 benign；最低扫描阈值时又把全部样本判为 malignant。这种输出不适合当前诊断工作流。

## 决策

- 不将该 ConvNeXt-Tiny checkpoint 加入 ensemble。
- 不为这一精确配置继续跑五折。
- 该结果不证明 ConvNeXt 永远不可用；它只说明 `convnext_tiny` 在当前 224 / CLAHE / horizontal-flip / lr=0.0003 recipe 下没有竞争力。
- 如果重新测试 ConvNeXt，应单独使用优化 recipe：更低 learning rate、best-epoch checkpoint selection、class balancing 和 probability calibration。

## Artifacts

- Config: `configs/classifier/convnext_tiny.yml`
- Training report: `artifacts/reports/train_cls_convnext_tiny_fold1.json`
- Checkpoint: `artifacts/checkpoints/convnext_tiny_fold1.pt`
- BUSI report: `artifacts/reports/busi_convnext_tiny_fold1.json`
- BUSI threshold analysis: `artifacts/reports/threshold_analysis_convnext_tiny_fold1.md`
