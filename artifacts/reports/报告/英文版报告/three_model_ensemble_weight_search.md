<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Three-Model Hybrid Ensemble Weight Search

Date: 2026-04-25

## Search Settings

- BUSI is only used for external evaluation and operating point analysis, not involved in any model training.
- This round searches weights on cached BUSI probabilities, so it does not reload large models for training.
- If BUSI is treated as the final unseen test set, this round's weight search results should be considered as candidate operating points on the external evaluation set, rather than fully unbiased generalization results.
- Sample count: `647`.
- Coarse search step: `0.01`; fine search step: `0.001`; fine search radius: `0.04`.
- Weights within each combination are normalized for direct comparison across single-model, two-model, and three-model schemes.

## Input Model Reports

- effnet: `artifacts\reports\busi_tta_fine_threshold_eval.json`
- densenet: `artifacts\reports\busi_densenet121_5fold_tta.json`
- convnext: `artifacts\reports\busi_convnext_tiny_tta_crop_sweep.json`

## Best AUC per Combination

| Combination | Weight | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | Optimal Threshold | Optimal Sens | Optimal Spec | Optimal Acc | Youden J | Precision | F1-Score | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| effnet | effnet 1.000 | 0.8997 | 0.6619 | 0.9382 | 0.8485 | 0.33 | 0.8000 | 0.8764 | 0.8516 | 0.6764 | 0.7568 | 0.7778 | TN 383 / FP 54 / FN 42 / TP 168 |
| densenet | densenet 1.000 | 0.8914 | 0.4952 | 0.9794 | 0.8223 | 0.19 | 0.7952 | 0.8581 | 0.8377 | 0.6534 | 0.7293 | 0.7608 | TN 375 / FP 62 / FN 43 / TP 167 |
| convnext | convnext 1.000 | 0.9054 | 0.7714 | 0.8879 | 0.8501 | 0.46 | 0.8048 | 0.8719 | 0.8501 | 0.6766 | 0.7511 | 0.7770 | TN 381 / FP 56 / FN 41 / TP 169 |
| effnet+densenet | effnet 0.612 / densenet 0.388 | 0.9053 | 0.5762 | 0.9565 | 0.8331 | 0.30 | 0.7810 | 0.9016 | 0.8624 | 0.6826 | 0.7923 | 0.7866 | TN 394 / FP 43 / FN 46 / TP 164 |
| effnet+convnext | effnet 0.427 / convnext 0.573 | 0.9133 | 0.7381 | 0.9153 | 0.8578 | 0.42 | 0.7857 | 0.8902 | 0.8563 | 0.6759 | 0.7746 | 0.7801 | TN 389 / FP 48 / FN 45 / TP 165 |
| densenet+convnext | densenet 0.396 / convnext 0.604 | 0.9082 | 0.7190 | 0.9268 | 0.8594 | 0.37 | 0.7857 | 0.8879 | 0.8547 | 0.6736 | 0.7710 | 0.7783 | TN 388 / FP 49 / FN 45 / TP 165 |
| effnet+densenet+convnext | effnet 0.358 / densenet 0.244 / convnext 0.398 | 0.9138 | 0.7095 | 0.9382 | 0.8640 | 0.32 | 0.8095 | 0.8627 | 0.8454 | 0.6722 | 0.7391 | 0.7727 | TN 377 / FP 60 / FN 40 / TP 170 |

## Best Youden Operating Point per Combination

| Combination | Weight | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | Optimal Threshold | Optimal Sens | Optimal Spec | Optimal Acc | Youden J | Precision | F1-Score | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| effnet | effnet 1.000 | 0.8997 | 0.6619 | 0.9382 | 0.8485 | 0.33 | 0.8000 | 0.8764 | 0.8516 | 0.6764 | 0.7568 | 0.7778 | TN 383 / FP 54 / FN 42 / TP 168 |
| densenet | densenet 1.000 | 0.8914 | 0.4952 | 0.9794 | 0.8223 | 0.19 | 0.7952 | 0.8581 | 0.8377 | 0.6534 | 0.7293 | 0.7608 | TN 375 / FP 62 / FN 43 / TP 167 |
| convnext | convnext 1.000 | 0.9054 | 0.7714 | 0.8879 | 0.8501 | 0.46 | 0.8048 | 0.8719 | 0.8501 | 0.6766 | 0.7511 | 0.7770 | TN 381 / FP 56 / FN 41 / TP 169 |
| effnet+densenet | effnet 0.713 / densenet 0.287 | 0.9047 | 0.6000 | 0.9451 | 0.8331 | 0.29 | 0.8095 | 0.8787 | 0.8563 | 0.6882 | 0.7623 | 0.7852 | TN 384 / FP 53 / FN 40 / TP 170 |
| effnet+convnext | effnet 0.760 / convnext 0.240 | 0.9094 | 0.6952 | 0.9405 | 0.8609 | 0.35 | 0.8048 | 0.8787 | 0.8547 | 0.6835 | 0.7613 | 0.7824 | TN 384 / FP 53 / FN 41 / TP 169 |
| densenet+convnext | densenet 0.255 / convnext 0.745 | 0.9080 | 0.7429 | 0.9130 | 0.8578 | 0.45 | 0.7810 | 0.9016 | 0.8624 | 0.6826 | 0.7923 | 0.7866 | TN 394 / FP 43 / FN 46 / TP 164 |
| effnet+densenet+convnext | effnet 0.623 / densenet 0.291 / convnext 0.086 | 0.9087 | 0.6286 | 0.9451 | 0.8423 | 0.31 | 0.8048 | 0.8947 | 0.8655 | 0.6995 | 0.7860 | 0.7953 | TN 391 / FP 46 / FN 41 / TP 169 |

## Overall Best

- Best AUC: `effnet+densenet+convnext`, weights `effnet 0.358 / densenet 0.244 / convnext 0.398`, AUC `0.9138`.
- Best Youden: `effnet+densenet+convnext`, weights `effnet 0.623 / densenet 0.291 / convnext 0.086`, Youden J `0.6995`.

## Top-Ranked Schemes by AUC

| Rank | Combination | Weight | AUC | Optimal Threshold | Optimal Sens | Precision | F1-Score | Optimal Spec | Optimal Acc | Youden J |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | effnet+densenet+convnext | effnet 0.358 / densenet 0.244 / convnext 0.398 | 0.9138 | 0.32 | 0.8095 | - | - | 0.8627 | 0.8454 | 0.6722 |
| 2 | effnet+convnext | effnet 0.427 / convnext 0.573 | 0.9133 | 0.42 | 0.7857 | - | - | 0.8902 | 0.8563 | 0.6759 |
| 3 | densenet+convnext | densenet 0.396 / convnext 0.604 | 0.9082 | 0.37 | 0.7857 | - | - | 0.8879 | 0.8547 | 0.6736 |
| 4 | convnext | convnext 1.000 | 0.9054 | 0.46 | 0.8048 | - | - | 0.8719 | 0.8501 | 0.6766 |
| 5 | effnet+densenet | effnet 0.612 / densenet 0.388 | 0.9053 | 0.30 | 0.7810 | - | - | 0.9016 | 0.8624 | 0.6826 |
| 6 | effnet | effnet 1.000 | 0.8997 | 0.33 | 0.8000 | - | - | 0.8764 | 0.8516 | 0.6764 |
| 7 | densenet | densenet 1.000 | 0.8914 | 0.19 | 0.7952 | - | - | 0.8581 | 0.8377 | 0.6534 |

## Conclusion

- The three-model ensemble achieves the highest AUC, indicating that ConvNeXt-Tiny is complementary to the existing EfficientNetV2-S and DenseNet121.
- If the goal is leaderboard-style AUC, prioritize the three-model best AUC weights; if the goal is a more balanced screening operating point, prioritize the best Youden weights.
- The current code's standard inference configuration is not yet suitable for directly writing three models into `demo.yml`, because ConvNeXt-Tiny uses timm mean/std, bicubic, and multi-crop TTA, whereas EfficientNetV2-S/DenseNet121 use their original preprocessing. Before deploying a three-model ensemble, per-member preprocessing configuration inference logic needs to be added.
