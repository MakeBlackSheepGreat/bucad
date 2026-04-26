<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# Tiny 推荐 Recipe 对比

日期：2026-04-25

## 目的

使用 timm-aware recipe 重新训练 `swin_tiny_patch4_window7_224` 和 `convnext_tiny`，并将它们的 BUSI 外部性能与当前两个 ensemble baselines 对比。

## Recipe 变化

| 项目 | Previous Quick Recipe | New timm-aware Recipe |
| --- | --- | --- |
| Pretrained normalization | `/255` only | timm mean/std normalization |
| Resize/interpolation | direct square resize | timm bicubic + crop_pct |
| Training schedule | fixed learning rate | warmup + cosine decay |
| Checkpoint | last checkpoint | best validation AUC checkpoint |
| Class imbalance | no explicit balancing | balanced class weights |
| Epochs | 20 | 30 |

该运行仍保留项目中的 ultrasound preprocessing 选择：CLAHE 和 horizontal flip augmentation。BUSI 仍只作为外部评估，不参与训练。

## BUSBRA Fold-1 验证

| Model | Recipe | Best Epoch | AUC | Sensitivity | Specificity | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Swin-Tiny | quick project recipe | 16 | 0.7950 | 0.8852 | 0.5020 | 0.6267 |
| Swin-Tiny | timm-aware recipe | 30 | 0.9053 | 0.7541 | 0.8933 | 0.8480 |
| ConvNeXt-Tiny | quick project recipe | 3 | 0.4519 | 0.0000 | 1.0000 | 0.6747 |
| ConvNeXt-Tiny | timm-aware recipe | 13 | 0.9259 | 0.7213 | 0.9328 | 0.8640 |

新 recipe 修复了明显的训练坍缩问题。ConvNeXt-Tiny 的 fold-1 AUC 从不可用的 0.4519 提升到 0.9259；Swin-Tiny 从 0.7950 提升到 0.9053。

## BUSI 外部评估：Default Threshold 0.50

| Model/System | AUC | Threshold | Sensitivity | Specificity | Accuracy | Confusion | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Swin-Tiny fold1 timm-aware | 0.8721 | 0.5000 | 0.7048 | 0.8902 | 0.8300 | TN 389 / FP 48 / FN 62 / TP 148 | single fold, no ensemble |
| ConvNeXt-Tiny fold1 timm-aware | 0.8926 | 0.5000 | 0.7762 | 0.8696 | 0.8393 | TN 380 / FP 57 / FN 47 / TP 163 | single fold, no ensemble |
| EfficientNetV2-S 5-fold ensemble + TTA | 0.8997 | 0.5000 | 0.6619 | 0.9382 | 0.8485 | TN 410 / FP 27 / FN 71 / TP 139 | current ensemble baseline |
| EfficientNetV2-S + DenseNet121 mixed ensemble | 0.9052 | 0.5000 | 0.5810 | 0.9542 | 0.8331 | TN 417 / FP 20 / FN 88 / TP 122 | current mixed ensemble baseline |

## BUSI 外部评估：Best Youden Operating Point

| Model/System | AUC | Threshold | Sensitivity | Specificity | Accuracy | Confusion | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Swin-Tiny fold1 timm-aware | 0.8721 | 0.1800 | 0.7524 | 0.8696 | 0.8315 | TN 380 / FP 57 / FN 52 / TP 158 | single fold, no ensemble |
| ConvNeXt-Tiny fold1 timm-aware | 0.8926 | 0.5400 | 0.7667 | 0.8879 | 0.8485 | TN 388 / FP 49 / FN 49 / TP 161 | single fold, no ensemble |
| EfficientNetV2-S 5-fold ensemble + TTA | 0.8997 | 0.3300 | 0.8000 | 0.8764 | 0.8516 | TN 383 / FP 54 / FN 42 / TP 168 | current ensemble baseline |
| EfficientNetV2-S + DenseNet121 mixed ensemble | 0.9052 | 0.2700 | 0.8095 | 0.8719 | 0.8516 | TN 381 / FP 56 / FN 40 / TP 170 | current mixed ensemble baseline |

## 结果解读

- 之前的 Tiny 结果不具代表性，因为 quick recipe 不符合 timm pretrained assumptions。
- ConvNeXt-Tiny 在 recipe 修复后成为可信候选。其单折 BUSI AUC 达到 0.8926，接近 EfficientNetV2-S 5-fold ensemble 的 0.8997。
- Swin-Tiny 也显著提升，但在该 recipe 下 BUSI AUC 低于 ConvNeXt-Tiny。
- 当前 mixed EfficientNetV2-S + DenseNet121 ensemble 在已比较系统中仍有最高 BUSI AUC：0.9052。
- 由于 Tiny 模型此时只有 single-fold checkpoints，公平下一步不是立即替换 final model，而是测试 ConvNeXt-Tiny 是否能通过五折或低权重 ensemble contribution 进一步提升。

## 决策

- 暂时保留当前 mixed ensemble 作为 runtime model。
- 将 ConvNeXt-Tiny timm-aware 视为严肃后续候选。
- 不再用旧 quick-recipe Tiny 结果作为最终模型判断依据。
- 如果训练时间允许，优先跑 ConvNeXt-Tiny 五折；Swin-Tiny 优先级更低。

## Artifacts

- Swin train config: `configs/classifier/swin_tiny_patch4_window7_224_timm_recipe.yml`
- ConvNeXt train config: `configs/classifier/convnext_tiny_timm_recipe.yml`
- Swin train report: `artifacts/reports/train_cls_swin_tiny_patch4_window7_224_timm_recipe_fold1.json`
- ConvNeXt train report: `artifacts/reports/train_cls_convnext_tiny_timm_recipe_fold1.json`
- Swin BUSI report: `artifacts/reports/busi_swin_tiny_timm_recipe_fold1.json`
- ConvNeXt BUSI report: `artifacts/reports/busi_convnext_tiny_timm_recipe_fold1.json`
