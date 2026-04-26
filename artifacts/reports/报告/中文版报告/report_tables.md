<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 报告可用表格

日期：2026-04-24

## EfficientNetV2-S 五折内部验证

| Fold | AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9248 | 0.6148 | - | - | 0.9565 | 0.8453 |
| 2 | 0.8988 | 0.7521 | - | - | 0.8386 | 0.8107 |
| 3 | 0.9077 | 0.6612 | - | - | 0.9016 | 0.8240 |
| 4 | 0.8891 | 0.6230 | - | - | 0.9289 | 0.8293 |
| 5 | 0.8523 | 0.6116 | - | - | 0.9016 | 0.8080 |

| Mean | AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNetV2-S | 0.8946 | 0.6525 | - | - | 0.9054 | 0.8235 |

## BUSI 外部评估

| Operating Point | AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Conventional metrics with flip TTA | 0.8997 | 0.6619 | - | - | 0.9382 | 0.8485 | 0.50 |
| Selected high-sensitivity point with flip TTA | 0.8997 | 0.8476 | - | - | 0.8078 | 0.8207 | 0.25 |
| High-sensitivity point with fine sweep | 0.8997 | 0.8524 | - | - | 0.8055 | 0.8207 | 0.24 |
| Best Youden point with fine sweep | 0.8997 | 0.8000 | - | - | 0.8764 | 0.8516 | 0.33 |
| Mixed EfficientNetV2-S + DenseNet121 ensemble | 0.9052 | 0.8095 | - | - | 0.8719 | 0.8516 | 0.27 |

## 剩余正式证据

| Task | Status |
| --- | --- |
| T059 EfficientNetV2-S 5-fold training | complete |
| T060 full model comparison | pending |

## 完整模型对比

| Rank | Model | AUC | Sensitivity | Precision | F1-Score | Specificity | Accuracy | Status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `tf_efficientnetv2_s` | 0.8937 | 0.7049 | - | - | 0.8775 | 0.8213 | completed |
| 2 | `densenet121` | 0.8867 | 0.7213 | - | - | 0.8775 | 0.8267 | completed |
| 3 | `resnet18` | 0.8756 | 0.7131 | - | - | 0.8577 | 0.8107 | completed |
| 4 | `mobilenetv3_small_100` | 0.8704 | 0.6557 | - | - | 0.9170 | 0.8320 | completed |
| 5 | `basic_cnn` | 0.6427 | 0.0000 | - | - | 0.9921 | 0.6693 | completed |
| 6 | `vgg16` | 0.5000 | 0.0000 | - | - | 1.0000 | 0.6747 | completed |
| - | `alexnet` | - | - | - | - | - | - | failed |
