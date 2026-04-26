<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# EfficientNetV2-S 五折训练总结

日期：2026-04-24

## 各折指标

| Fold | AUC | Sensitivity | Specificity | Accuracy | Checkpoint |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.9248 | 0.6148 | 0.9565 | 0.8453 | `C:/Users/876762330/Desktop/Agent/artifacts/checkpoints/efficientnetv2_s_fold1.pt` |
| 2 | 0.8988 | 0.7521 | 0.8386 | 0.8107 | `C:/Users/876762330/Desktop/Agent/artifacts/checkpoints/efficientnetv2_s_fold2.pt` |
| 3 | 0.9077 | 0.6612 | 0.9016 | 0.8240 | `C:/Users/876762330/Desktop/Agent/artifacts/checkpoints/efficientnetv2_s_fold3.pt` |
| 4 | 0.8891 | 0.6230 | 0.9289 | 0.8293 | `C:/Users/876762330/Desktop/Agent/artifacts/checkpoints/efficientnetv2_s_fold4.pt` |
| 5 | 0.8523 | 0.6116 | 0.9016 | 0.8080 | `C:/Users/876762330/Desktop/Agent/artifacts/checkpoints/efficientnetv2_s_fold5.pt` |

## 内部验证均值

- Mean AUC：`0.8946`
- Mean Sensitivity：`0.6525`
- Mean Specificity：`0.9054`
- Mean Accuracy：`0.8235`

## 五折集成的 BUSI 外部评估

- AUC：`0.8955`
- 默认 0.50 阈值下 Sensitivity / Specificity / Accuracy：`0.6667` / `0.9382` / `0.8501`
- Best Youden threshold：`0.25`
- 最优阈值下 Sensitivity / Specificity / Accuracy：`0.8476` / `0.8215` / `0.8300`

## 决策影响

- T059 已完成：五个 EfficientNetV2-S fold checkpoint 和报告均已存在。
- Runtime inference 已在 `configs/inference/demo.yml` 中切换到 EfficientNetV2-S 五折集成。
- AUC 目标已超过。使用 Youden 最优阈值 `0.25` 时，Sensitivity 基本接近 0.85 目标，但仍差 `0.0024`。
