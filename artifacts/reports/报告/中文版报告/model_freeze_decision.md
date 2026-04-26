<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 模型与阈值冻结决策

日期：2026-04-24

## 决策

- 冻结 runtime classifier：EfficientNetV2-S 五折 + DenseNet121 五折 mixed ensemble。
- Runtime checkpoints 包括五个 `efficientnetv2_s_fold*.pt` 和五个 `densenet121_fold*.pt`。
- 冻结 runtime segmenter：`artifacts/checkpoints/segmenter_fold1.pt`。
- Runtime inference enhancement：`configs/inference/demo.yml` 中启用 horizontal-flip TTA。
- UI/runtime balanced threshold：`0.27`。
- 评估说明：`evaluate_busi_dataset` 仍将常规 0.50 指标写入 `metrics`；选定运行点记录在 `threshold_analysis.best_by_youden`。

## 证据

| Evidence | File | Key Result |
| --- | --- | --- |
| EfficientNetV2-S 5-fold summary | `artifacts/reports/efficientnetv2_s_5fold_summary.md` | mean AUC=0.8946，mean sensitivity=0.6525，mean specificity=0.9054 |
| BUSI mixed-ensemble evaluation | `artifacts/reports/busi_mixed_ensemble_weight063_eval.json` | AUC=0.9052；threshold 0.27 下 sensitivity=0.8095，specificity=0.8719，accuracy=0.8516 |
| Release config | `configs/inference/demo.yml` | EfficientNetV2-S + DenseNet121 mixed ensemble，flip TTA，threshold 0.27 |

## 目标差距

- AUC 目标 >= 0.75：PASS，BUSI 上超过 `+0.1552`。
- Sensitivity 目标 0.85：当前 balanced runtime point 优先考虑 AUC 和 Specificity；若需要高 Sensitivity 运行点，可使用 threshold 0.24。
- 选定 balanced threshold 下 Specificity 为 0.8719。

## 局限

- 当时 T060 完整模型对比仍待完成，因此最终报告应表述为 EfficientNetV2-S 是 selected/frozen model，而不是已证明优于所有 baseline。
- BUSI AUC 通过 mixed ensemble 提升，但系统仍是 research prototype，不是 clinical software。
- Flip TTA 提升 AUC，但相较此前 non-TTA 评估，在 threshold 0.25 下 Specificity 略低。
- Grad-CAM 和 segmentation 是解释辅助，不是临床 ground truth。

## 完整对比结果

- T060 已用 fold 1、20 epochs 完成 7 个配置模型对比。
- 按 AUC 排名的最佳完成模型：`tf_efficientnetv2_s`，AUC `0.8937`。
- 强 baseline：`densenet121` AUC `0.8867`，`resnet18` AUC `0.8756`，`mobilenetv3_small_100` AUC `0.8704`。
- `alexnet` 在记录运行中失败，并已记录在 `artifacts/reports/comparison_results.json`；之后为未来重跑加入了 torchvision fallback。
- 结论：已完成的对比支持 EfficientNetV2-S freeze decision。
