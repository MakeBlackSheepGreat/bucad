# 四个五折单模型性能对比

> 数据边界说明：BUSBRA 用于训练、内部验证、OOF、模型选择、阈值选择和 ROI 参数选择；BUSI 是锁定外部评估集，只用于最终外部验证，不用于训练或调参。


日期：2026-04-25

## 说明

- 本报告只比较单一模型家族的五折结果，不包含跨模型混合集成。
- BUSBRA 用于训练和内部验证；BUSI 只用于外部评估和运行点分析。
- ConvNeXt-Tiny 与 Swin-Tiny 使用各自 timm-aware 预处理；EfficientNetV2-S 与 DenseNet121 使用原有 224 + CLAHE + hflip TTA 协议。

## BUSBRA 内部五折均值

| 模型 | Mean AUC | Mean Sens | Precision | F1-Score | Mean Spec | Mean Acc | 协议 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EfficientNetV2-S 5-fold + hflip TTA | 0.8946 | 0.6525 | - | - | 0.9054 | 0.8235 | 224 + CLAHE + hflip TTA |
| DenseNet121 5-fold + hflip TTA | 0.8749 | 0.6489 | - | - | 0.9148 | 0.8288 | 224 + CLAHE + hflip TTA |
| ConvNeXt-Tiny 5-fold + crop_sweep TTA | 0.9212 | 0.7744 | - | - | 0.8864 | 0.8501 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.9139 | 0.7595 | - | - | 0.8872 | 0.8459 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |

## BUSI 外部评估：默认阈值 0.50

| 模型 | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | Precision | F1-Score | Confusion | 协议 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ConvNeXt-Tiny 5-fold + crop_sweep TTA | 0.9054 | 0.7714 | 0.8879 | 0.8501 | 0.7678 | 0.7696 | TN 388 / FP 49 / FN 48 / TP 162 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |
| EfficientNetV2-S 5-fold + hflip TTA | 0.8997 | 0.6619 | 0.9382 | 0.8485 | 0.8373 | 0.7394 | TN 410 / FP 27 / FN 71 / TP 139 | 224 + CLAHE + hflip TTA |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.8971 | 0.7762 | 0.8833 | 0.8485 | 0.7617 | 0.7689 | TN 386 / FP 51 / FN 47 / TP 163 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |
| DenseNet121 5-fold + hflip TTA | 0.8914 | 0.4952 | 0.9794 | 0.8223 | 0.9204 | 0.6440 | TN 428 / FP 9 / FN 106 / TP 104 | 224 + CLAHE + hflip TTA |

## BUSI 外部评估：Youden 最优运行点

| 模型 | AUC | 阈值 | Sens | Spec | Acc | Youden J | Precision | F1-Score | Confusion | 协议 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ConvNeXt-Tiny 5-fold + crop_sweep TTA | 0.9054 | 0.46 | 0.8048 | 0.8719 | 0.8501 | 0.6766 | 0.7511 | 0.7770 | TN 381 / FP 56 / FN 41 / TP 169 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |
| EfficientNetV2-S 5-fold + hflip TTA | 0.8997 | 0.33 | 0.8000 | 0.8764 | 0.8516 | 0.6764 | 0.7568 | 0.7778 | TN 383 / FP 54 / FN 42 / TP 168 | 224 + CLAHE + hflip TTA |
| Swin-Tiny 5-fold + crop_sweep TTA | 0.8971 | 0.50 | 0.7762 | 0.8833 | 0.8485 | 0.6595 | 0.7617 | 0.7689 | TN 386 / FP 51 / FN 47 / TP 163 | 224 + CLAHE + timm mean/std + bicubic + crop_sweep TTA |
| DenseNet121 5-fold + hflip TTA | 0.8914 | 0.19 | 0.7952 | 0.8581 | 0.8377 | 0.6534 | 0.7293 | 0.7608 | TN 375 / FP 62 / FN 43 / TP 167 | 224 + CLAHE + hflip TTA |

## 结论

- 五折单模型里，BUSI AUC 最高的是 ConvNeXt-Tiny 5-fold + crop_sweep TTA，AUC `0.9054`。
- Youden J 最高的是 ConvNeXt-Tiny 5-fold + crop_sweep TTA，Youden J `0.6766`。
- ConvNeXt-Tiny 是当前最强的单模型五折候选；Swin-Tiny 五折虽然比 fold1 有提升，但仍低于 ConvNeXt-Tiny。
