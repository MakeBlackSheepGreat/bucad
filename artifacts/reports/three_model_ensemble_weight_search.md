# 三模型混合集成权重搜索

日期：2026-04-25

## 搜索设置

- BUSI 只用于外部评估和运行点分析，不参与任何模型训练。
- 本轮是在已缓存的 BUSI 概率上搜索权重，因此不会重复加载大模型训练。
- 如果把 BUSI 当作最终未知测试集，本轮权重搜索结果应视为外部评估集上的候选运行点，而不是完全无偏的泛化结果。
- 样本数：`647`。
- 粗搜步长：`0.01`；细搜步长：`0.001`；细搜半径：`0.04`。
- 每一种组合内部的权重都会归一化，便于直接比较单模型、两模型和三模型方案。

## 输入模型报告

- effnet: `artifacts\reports\busi_tta_fine_threshold_eval.json`
- densenet: `artifacts\reports\busi_densenet121_5fold_tta.json`
- convnext: `artifacts\reports\busi_convnext_tiny_tta_crop_sweep.json`

## 各组合的最佳 AUC

| 组合 | 权重 | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | 最优阈值 | 最优 Sens | 最优 Spec | 最优 Acc | Youden J | 混淆矩阵 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| effnet | effnet 1.000 | 0.8997 | 0.6619 | 0.9382 | 0.8485 | 0.33 | 0.8000 | 0.8764 | 0.8516 | 0.6764 | TN 383 / FP 54 / FN 42 / TP 168 |
| densenet | densenet 1.000 | 0.8914 | 0.4952 | 0.9794 | 0.8223 | 0.19 | 0.7952 | 0.8581 | 0.8377 | 0.6534 | TN 375 / FP 62 / FN 43 / TP 167 |
| convnext | convnext 1.000 | 0.9054 | 0.7714 | 0.8879 | 0.8501 | 0.46 | 0.8048 | 0.8719 | 0.8501 | 0.6766 | TN 381 / FP 56 / FN 41 / TP 169 |
| effnet+densenet | effnet 0.612 / densenet 0.388 | 0.9053 | 0.5762 | 0.9565 | 0.8331 | 0.30 | 0.7810 | 0.9016 | 0.8624 | 0.6826 | TN 394 / FP 43 / FN 46 / TP 164 |
| effnet+convnext | effnet 0.427 / convnext 0.573 | 0.9133 | 0.7381 | 0.9153 | 0.8578 | 0.42 | 0.7857 | 0.8902 | 0.8563 | 0.6759 | TN 389 / FP 48 / FN 45 / TP 165 |
| densenet+convnext | densenet 0.396 / convnext 0.604 | 0.9082 | 0.7190 | 0.9268 | 0.8594 | 0.37 | 0.7857 | 0.8879 | 0.8547 | 0.6736 | TN 388 / FP 49 / FN 45 / TP 165 |
| effnet+densenet+convnext | effnet 0.358 / densenet 0.244 / convnext 0.398 | 0.9138 | 0.7095 | 0.9382 | 0.8640 | 0.32 | 0.8095 | 0.8627 | 0.8454 | 0.6722 | TN 377 / FP 60 / FN 40 / TP 170 |

## 各组合的最佳 Youden 运行点

| 组合 | 权重 | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | 最优阈值 | 最优 Sens | 最优 Spec | 最优 Acc | Youden J | 混淆矩阵 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| effnet | effnet 1.000 | 0.8997 | 0.6619 | 0.9382 | 0.8485 | 0.33 | 0.8000 | 0.8764 | 0.8516 | 0.6764 | TN 383 / FP 54 / FN 42 / TP 168 |
| densenet | densenet 1.000 | 0.8914 | 0.4952 | 0.9794 | 0.8223 | 0.19 | 0.7952 | 0.8581 | 0.8377 | 0.6534 | TN 375 / FP 62 / FN 43 / TP 167 |
| convnext | convnext 1.000 | 0.9054 | 0.7714 | 0.8879 | 0.8501 | 0.46 | 0.8048 | 0.8719 | 0.8501 | 0.6766 | TN 381 / FP 56 / FN 41 / TP 169 |
| effnet+densenet | effnet 0.713 / densenet 0.287 | 0.9047 | 0.6000 | 0.9451 | 0.8331 | 0.29 | 0.8095 | 0.8787 | 0.8563 | 0.6882 | TN 384 / FP 53 / FN 40 / TP 170 |
| effnet+convnext | effnet 0.760 / convnext 0.240 | 0.9094 | 0.6952 | 0.9405 | 0.8609 | 0.35 | 0.8048 | 0.8787 | 0.8547 | 0.6835 | TN 384 / FP 53 / FN 41 / TP 169 |
| densenet+convnext | densenet 0.255 / convnext 0.745 | 0.9080 | 0.7429 | 0.9130 | 0.8578 | 0.45 | 0.7810 | 0.9016 | 0.8624 | 0.6826 | TN 394 / FP 43 / FN 46 / TP 164 |
| effnet+densenet+convnext | effnet 0.623 / densenet 0.291 / convnext 0.086 | 0.9087 | 0.6286 | 0.9451 | 0.8423 | 0.31 | 0.8048 | 0.8947 | 0.8655 | 0.6995 | TN 391 / FP 46 / FN 41 / TP 169 |

## 总体最佳

- 最佳 AUC：`effnet+densenet+convnext`，权重 `effnet 0.358 / densenet 0.244 / convnext 0.398`，AUC `0.9138`。
- 最佳 Youden：`effnet+densenet+convnext`，权重 `effnet 0.623 / densenet 0.291 / convnext 0.086`，Youden J `0.6995`。

## AUC 排名前列方案

| 排名 | 组合 | 权重 | AUC | 最优阈值 | 最优 Sens | 最优 Spec | 最优 Acc | Youden J |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | effnet+densenet+convnext | effnet 0.358 / densenet 0.244 / convnext 0.398 | 0.9138 | 0.32 | 0.8095 | 0.8627 | 0.8454 | 0.6722 |
| 2 | effnet+convnext | effnet 0.427 / convnext 0.573 | 0.9133 | 0.42 | 0.7857 | 0.8902 | 0.8563 | 0.6759 |
| 3 | densenet+convnext | densenet 0.396 / convnext 0.604 | 0.9082 | 0.37 | 0.7857 | 0.8879 | 0.8547 | 0.6736 |
| 4 | convnext | convnext 1.000 | 0.9054 | 0.46 | 0.8048 | 0.8719 | 0.8501 | 0.6766 |
| 5 | effnet+densenet | effnet 0.612 / densenet 0.388 | 0.9053 | 0.30 | 0.7810 | 0.9016 | 0.8624 | 0.6826 |
| 6 | effnet | effnet 1.000 | 0.8997 | 0.33 | 0.8000 | 0.8764 | 0.8516 | 0.6764 |
| 7 | densenet | densenet 1.000 | 0.8914 | 0.19 | 0.7952 | 0.8581 | 0.8377 | 0.6534 |

## 结论

- 三模型一起集成取得最高 AUC，说明 ConvNeXt-Tiny 与原来的 EfficientNetV2-S、DenseNet121 存在互补性。
- 如果目标是排行榜式 AUC，优先选择三模型最佳 AUC 权重；如果目标是更均衡的筛查运行点，优先看最佳 Youden 权重。
- 当前代码的普通推理配置还不适合直接把三模型写进 `demo.yml`，因为 ConvNeXt-Tiny 使用 timm mean/std、bicubic 和多裁剪 TTA，而 EfficientNetV2-S/DenseNet121 使用原有预处理。真正部署三模型集成前，需要增加按成员配置预处理的推理逻辑。
