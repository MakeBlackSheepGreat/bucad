<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# ConvNeXt-Tiny 主模型优化

日期：2026-04-25

## 目标

将 ConvNeXt-Tiny 作为潜在新主分类器进行优化，并与当前两个 ensemble baseline 对比。BUSBRA 仍然是唯一的训练和内部验证数据来源，BUSI 仅作为外部评估集。

## 训练 Recipe

| 项目 | 设置 |
| --- | --- |
| Backbone | `convnext_tiny` |
| Pretraining | timm pretrained weights |
| Input | 224，timm bicubic interpolation，crop_pct 0.95 |
| Normalization | timm mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| Ultrasound preprocessing | CLAHE enabled |
| Augmentation | horizontal flip |
| Optimizer | AdamW |
| Schedule | warmup + cosine decay |
| Class balance | balanced class weights |
| Checkpoint selection | best validation AUC |
| Folds | 5 |

## BUSBRA 五折内部验证

| Fold | Best Epoch | AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13 | 0.9259 | 0.7213 | - | - | 0.9328 | 0.8640 |
| 2 | 3 | 0.9301 | 0.9008 | - | - | 0.7913 | 0.8267 |
| 3 | 30 | 0.9469 | 0.8099 | - | - | 0.9173 | 0.8827 |
| 4 | 14 | 0.8875 | 0.7623 | - | - | 0.8577 | 0.8267 |
| 5 | 10 | 0.9158 | 0.6777 | - | - | 0.9331 | 0.8507 |
| Mean | - | 0.9212 | 0.7744 | - | - | 0.8864 | 0.8501 |

五折内部结果已经足够稳定，可以把 ConvNeXt-Tiny 视为严肃的主模型候选。fold 4 弱于其他 folds，但 mean AUC 仍保持在 0.91 以上。

## BUSI 外部评估：Default Threshold 0.50

| Model/System | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ConvNeXt-Tiny fold1 | 0.8926 | 0.5000 | 0.7762 | 0.8696 | 0.8393 | 0.7409 | 0.7581 | TN 380 / FP 57 / FN 47 / TP 163 | single fold |
| ConvNeXt-Tiny 5-fold ensemble | 0.9044 | 0.5000 | 0.7524 | 0.8947 | 0.8485 | 0.7745 | 0.7633 | TN 391 / FP 46 / FN 52 / TP 158 | new main-model candidate |
| EfficientNetV2-S 5-fold + TTA | 0.8997 | 0.5000 | 0.6619 | 0.9382 | 0.8485 | 0.8373 | 0.7394 | TN 410 / FP 27 / FN 71 / TP 139 | existing ensemble baseline |
| EfficientNetV2-S + DenseNet121 mixed ensemble | 0.9052 | 0.5000 | 0.5810 | 0.9542 | 0.8331 | 0.8592 | 0.6932 | TN 417 / FP 20 / FN 88 / TP 122 | current runtime baseline |

## BUSI 外部评估：Best Youden Operating Point

| Model/System | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ConvNeXt-Tiny fold1 | 0.8926 | 0.5400 | 0.7667 | 0.8879 | 0.8485 | 0.7667 | 0.7667 | TN 388 / FP 49 / FN 49 / TP 161 | single fold |
| ConvNeXt-Tiny 5-fold ensemble | 0.9044 | 0.4400 | 0.7762 | 0.8741 | 0.8423 | 0.7477 | 0.7617 | TN 382 / FP 55 / FN 47 / TP 163 | new main-model candidate |
| EfficientNetV2-S 5-fold + TTA | 0.8997 | 0.3300 | 0.8000 | 0.8764 | 0.8516 | 0.7568 | 0.7778 | TN 383 / FP 54 / FN 42 / TP 168 | existing ensemble baseline |
| EfficientNetV2-S + DenseNet121 mixed ensemble | 0.9052 | 0.2700 | 0.8095 | 0.8719 | 0.8516 | 0.7522 | 0.7798 | TN 381 / FP 56 / FN 40 / TP 170 | current runtime baseline |

## 结果解读

- ConvNeXt-Tiny 5-fold 将 BUSI AUC 从单折的 `0.8926` 提升到 `0.9044`。
- ConvNeXt-Tiny 5-fold 的 AUC 略高于 EfficientNetV2-S 5-fold + TTA baseline：`0.9044` vs `0.8997`。
- 当前 EfficientNetV2-S + DenseNet121 mixed ensemble 的 AUC 仍略高：`0.9052` vs `0.9044`。
- ConvNeXt-Tiny 在 Specificity/Accuracy 平衡上表现强，但 best-Youden Sensitivity 仍低于 mixed ensemble。
- 由于 ConvNeXt-Tiny 已接近当前 runtime baseline，它可以作为下一轮 ensemble 搜索的主模型候选。

## 下一步 Ensemble 方向

- 暂不替换 runtime model；mixed EfficientNetV2-S + DenseNet121 ensemble 仍有最高 AUC。
- 低权重 Swin-Tiny 搜索应在 Swin-Tiny 完成五折后再做，或者明确标注为单折探索成员。
- 更直接的方向是 ConvNeXt-Tiny 5-fold + 当前 mixed ensemble 权重搜索，因为两侧都有五折证据。
- 如果目标是构建干净的新模型族 ensemble，应先用 timm-aware recipe 训练 Swin-Tiny folds 2-5，再搜索 ConvNeXt:Swin 权重。

## Artifacts

- Train config: `configs/classifier/convnext_tiny_timm_recipe.yml`
- Inference config: `configs/inference/convnext_tiny_timm_recipe_5fold.yml`
- BUSI report: `artifacts/reports/busi_convnext_tiny_timm_recipe_5fold.json`
- Threshold report: `artifacts/reports/threshold_analysis_convnext_tiny_timm_recipe_5fold.md`
