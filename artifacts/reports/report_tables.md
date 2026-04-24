# Report-Ready Tables

Date: 2026-04-24

## EfficientNetV2-S 5-Fold Internal Validation

| Fold | AUC | Sensitivity | Specificity | Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9248 | 0.6148 | 0.9565 | 0.8453 |
| 2 | 0.8988 | 0.7521 | 0.8386 | 0.8107 |
| 3 | 0.9077 | 0.6612 | 0.9016 | 0.8240 |
| 4 | 0.8891 | 0.6230 | 0.9289 | 0.8293 |
| 5 | 0.8523 | 0.6116 | 0.9016 | 0.8080 |

| Mean | AUC | Sensitivity | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| EfficientNetV2-S | 0.8946 | 0.6525 | 0.9054 | 0.8235 |

## BUSI External Evaluation

| Operating Point | AUC | Sensitivity | Specificity | Accuracy | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional metrics | 0.8955 | 0.6667 | 0.9382 | 0.8501 | 0.50 |
| Selected high-sensitivity point | 0.8955 | 0.8476 | 0.8215 | 0.8300 | 0.25 |

## Remaining Formal Evidence

| Task | Status |
| --- | --- |
| T059 EfficientNetV2-S 5-fold training | complete |
| T060 full model comparison | pending |
## Full Model Comparison

| Rank | Model | AUC | Sensitivity | Specificity | Accuracy | Status |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `tf_efficientnetv2_s` | 0.8937 | 0.7049 | 0.8775 | 0.8213 | completed |
| 2 | `densenet121` | 0.8867 | 0.7213 | 0.8775 | 0.8267 | completed |
| 3 | `resnet18` | 0.8756 | 0.7131 | 0.8577 | 0.8107 | completed |
| 4 | `mobilenetv3_small_100` | 0.8704 | 0.6557 | 0.9170 | 0.8320 | completed |
| 5 | `basic_cnn` | 0.6427 | 0.0000 | 0.9921 | 0.6693 | completed |
| 6 | `vgg16` | 0.5000 | 0.0000 | 1.0000 | 0.6747 | completed |
| - | `alexnet` | - | - | - | - | failed |
