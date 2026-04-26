<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# Swin-Tiny Transformer 实验

日期：2026-04-25

## 目标

评估轻量 Transformer backbone 是否能在投入五折训练或 ensemble integration 之前，改善当前乳腺超声分类器 baseline。

## 实验设置

| 项目 | 设置 |
| --- | --- |
| Model | `swin_tiny_patch4_window7_224` |
| Training data | BUSBRA only |
| Fold | 1 |
| Epochs | 20 |
| Input size | 224 |
| Preprocessing | CLAHE |
| Augmentation | Horizontal flip |
| Learning rate | 0.0001 |
| BUSI role | 仅作外部评估，不参与训练 |

## Recipe 局限

这次实验是快速 project-recipe 初筛，不是完整 Swin 专项优化。实验复用了当前分类器流水线：square resize、CLAHE、horizontal flip、AdamW、固定 learning rate、无 scheduler/warmup，并保存 last checkpoint。

该配置没有使用 timm 针对 Swin 的完整 pretrained transform 和训练 recipe，例如 ImageNet mean/std normalization、crop policy 或 Swin-specific fine-tuning choices。

因此，本实验可以否定这一精确快速 recipe，但不能用于断言 Swin Transformer 天然不适合乳腺超声分类。

## BUSBRA Fold-1 验证

该表是 fold-1 单模型对比，应用于判断 comparable validation split 下的 backbone 本身，不应拿来和最终 optimized ensemble 对比。

| Model | AUC | Sensitivity | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| EfficientNetV2-S fold1 comparison run | 0.8937 | 0.7049 | 0.8775 | 0.8213 |
| DenseNet121 fold1 | 0.8867 | 0.7213 | 0.8775 | 0.8267 |
| Swin-Tiny fold1 | 0.7950 | 0.8852 | 0.5020 | 0.6267 |

Swin-Tiny 在默认阈值下显著提高了 Sensitivity，但提升来自于把更多样本判为 malignant。Specificity 和 Accuracy 大幅下降，AUC 也明显低于 fold-1 EfficientNetV2-S 和 DenseNet121 单模型 baseline。

## BUSI 外部评估

| Operating Point | AUC | Threshold | Sensitivity | Specificity | Accuracy | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default threshold | 0.8223 | 0.50 | 0.8810 | 0.5584 | 0.6631 | TN 244 / FP 193 / FN 25 / TP 185 |
| Best Youden point | 0.8223 | 0.57 | 0.7619 | 0.7483 | 0.7527 | TN 327 / FP 110 / FN 50 / TP 160 |

这里 BUSI 只作为该单折模型的外部 sanity check。默认阈值能得到较高 Sensitivity，但 false positives 太多。这个证据不足以支持为该精确 Swin-Tiny setup 投入五折训练时间。

## 决策

- 基于 fold-1 单模型结果，不为该精确 Swin-Tiny 配置投入五折训练时间。
- 如果仍希望探索 Transformer-style 模型，可以转向 hybrid 或 modern CNN 候选，例如 `convnext_tiny`、`maxvit_tiny_rw_224` 或 `coatnet_0_rw_224`。
- 对于 Sensitivity 提升，当前证据更支持 threshold tuning、balanced-sensitive EfficientNet training 或 heterogeneous ensemble refinement，而不是该 Swin-Tiny 配置。

## Artifacts

- Config: `configs/classifier/swin_tiny_patch4_window7_224.yml`
- Training report: `artifacts/reports/train_cls_swin_tiny_patch4_window7_224_fold1.json`
- Checkpoint: `artifacts/checkpoints/swin_tiny_patch4_window7_224_fold1.pt`
- BUSI report: `artifacts/reports/busi_swin_tiny_patch4_window7_224_fold1.json`
- BUSI threshold analysis: `artifacts/reports/threshold_analysis_swin_tiny_patch4_window7_224_fold1.md`
