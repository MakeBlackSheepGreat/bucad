<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 全模型对比总结

日期：2026-04-24

命令：`conda run -n BUCAD python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20`

Dry run：`False`
模型数量：`7`

## 结果

| 排名 | 模型 | 状态 | AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy | 运行秒数 | 备注 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `tf_efficientnetv2_s` | completed | 0.8937 | 0.7049 | - | - | 0.8775 | 0.8213 | 342.1 | EfficientNetV2-S 主模型 |
| 2 | `densenet121` | completed | 0.8867 | 0.7213 | - | - | 0.8775 | 0.8267 | 638.4 | DenseNet-121 baseline |
| 3 | `resnet18` | completed | 0.8756 | 0.7131 | - | - | 0.8577 | 0.8107 | 153.9 | ResNet-18 baseline |
| 4 | `mobilenetv3_small_100` | completed | 0.8704 | 0.6557 | - | - | 0.9170 | 0.8320 | 188.2 | MobileNetV3-Small baseline |
| 5 | `basic_cnn` | completed | 0.6427 | 0.0000 | - | - | 0.9921 | 0.6693 | 90.6 | 无预训练权重的 Basic CNN baseline |
| 6 | `vgg16` | completed | 0.5000 | 0.0000 | - | - | 1.0000 | 0.6747 | 707.2 | VGG-16 baseline |
| - | `alexnet` | failed | - | - | - | - | - | - | 0.3 | 记录运行时模型工厂尚不支持 AlexNet；之后已加入 torchvision fallback |

## 结论

- 在已完成的 fold-1 对比模型中，`tf_efficientnetv2_s` 的验证 AUC 最高。
- `densenet121`、`resnet18` 和 `mobilenetv3_small_100` 都是较强的次级 baseline。
- `alexnet` 在本次记录运行中失败；如果后续需要 AlexNet 指标，应在已加入 fallback 支持后重新运行。
- 本次对比支持将 EfficientNetV2-S 冻结为当时的最终分类器候选家族。

## 外部评估背景

- BUSI ensemble AUC：`0.8955`
- 选定阈值：`0.25`
- 选定阈值下 Sensitivity / Specificity / Accuracy：`0.8476` / `0.8215` / `0.8300`
