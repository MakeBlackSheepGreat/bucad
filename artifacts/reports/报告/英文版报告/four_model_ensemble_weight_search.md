<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Four Five-Fold Model Hybrid Ensemble Weight Fine Search

> Data boundary statement: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or hyperparameter tuning.


Date: 2026-04-25

## Search Settings

- Input models: EfficientNetV2-S five-fold, DenseNet121 five-fold, ConvNeXt-Tiny five-fold, Swin-Tiny five-fold.
- Search scope: all pairwise combinations, all three-way combinations, all four models together; single models are listed only as references and do not enter the main table of this report.
- Search method: weight normalization; coarse search step `0.01`, then `0.001` fine search around the best points for AUC, Youden, and Accuracy.
- BUSI is only used for external evaluation and operating point analysis, not involved in training.
- Sample count: `647`; number of combinations: `11`.

## Best AUC per Combination

| Combination | Weight | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | Optimal Threshold | Optimal Sens | Optimal Spec | Optimal Acc | Youden J | Precision | F1-Score | Confusion Matrix | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| convnext+swin | convnext 0.828 / swin 0.172 | 0.9071 | 0.7667 | 0.8970 | 0.8547 | 0.45 | 0.7952 | 0.8764 | 0.8501 | 0.6717 | 0.7557 | 0.7749 | TN 383 / FP 54 / FN 43 / TP 167 |  |
| densenet+convnext | densenet 0.396 / convnext 0.604 | 0.9082 | 0.7190 | 0.9268 | 0.8594 | 0.37 | 0.7857 | 0.8879 | 0.8547 | 0.6736 | 0.7710 | 0.7783 | TN 388 / FP 49 / FN 45 / TP 165 |  |
| densenet+swin | densenet 0.518 / swin 0.482 | 0.9019 | 0.7095 | 0.9336 | 0.8609 | 0.45 | 0.7429 | 0.9268 | 0.8671 | 0.6696 | 0.8298 | 0.7839 | TN 405 / FP 32 / FN 54 / TP 156 |  |
| effnet+convnext | effnet 0.427 / convnext 0.573 | 0.9133 | 0.7381 | 0.9153 | 0.8578 | 0.42 | 0.7857 | 0.8902 | 0.8563 | 0.6759 | 0.7746 | 0.7801 | TN 389 / FP 48 / FN 45 / TP 165 |  |
| effnet+densenet | effnet 0.612 / densenet 0.388 | 0.9053 | 0.5762 | 0.9565 | 0.8331 | 0.30 | 0.7810 | 0.9016 | 0.8624 | 0.6826 | 0.7923 | 0.7866 | TN 394 / FP 43 / FN 46 / TP 164 |  |
| effnet+swin | effnet 0.630 / swin 0.370 | 0.9075 | 0.7333 | 0.9291 | 0.8655 | 0.34 | 0.8286 | 0.8696 | 0.8563 | 0.6981 | 0.7532 | 0.7891 | TN 380 / FP 57 / FN 36 / TP 174 |  |
| densenet+convnext+swin | densenet 0.289 / convnext 0.501 / swin 0.210 | 0.9086 | 0.7429 | 0.9199 | 0.8624 | 0.48 | 0.7619 | 0.9176 | 0.8671 | 0.6795 | 0.8163 | 0.7882 | TN 401 / FP 36 / FN 50 / TP 160 |  |
| effnet+convnext+swin | effnet 0.437 / convnext 0.551 / swin 0.012 | 0.9134 | 0.7333 | 0.9176 | 0.8578 | 0.25 | 0.9000 | 0.7803 | 0.8192 | 0.6803 | 0.6632 | 0.7636 | TN 341 / FP 96 / FN 21 / TP 189 |  |
| effnet+densenet+convnext | effnet 0.358 / densenet 0.244 / convnext 0.398 | 0.9138 | 0.7095 | 0.9382 | 0.8640 | 0.32 | 0.8095 | 0.8627 | 0.8454 | 0.6722 | 0.7391 | 0.7727 | TN 377 / FP 60 / FN 40 / TP 170 |  |
| effnet+densenet+swin | effnet 0.434 / densenet 0.305 / swin 0.261 | 0.9091 | 0.6952 | 0.9474 | 0.8655 | 0.29 | 0.8238 | 0.8627 | 0.8501 | 0.6865 | 0.7425 | 0.7810 | TN 377 / FP 60 / FN 37 / TP 173 |  |
| effnet+densenet+convnext+swin | effnet 0.358 / densenet 0.244 / convnext 0.398 / swin 0.000 | 0.9138 | 0.7095 | 0.9382 | 0.8640 | 0.32 | 0.8095 | 0.8627 | 0.8454 | 0.6722 | 0.7391 | 0.7727 | TN 377 / FP 60 / FN 40 / TP 170 |  |

## Best Youden Operating Point per Combination

| Combination | Weight | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | Optimal Threshold | Optimal Sens | Optimal Spec | Optimal Acc | Youden J | Precision | F1-Score | Confusion Matrix | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| convnext+swin | convnext 0.902 / swin 0.098 | 0.9066 | 0.7667 | 0.8924 | 0.8516 | 0.43 | 0.8143 | 0.8650 | 0.8485 | 0.6793 | 0.7435 | 0.7773 | TN 378 / FP 59 / FN 39 / TP 171 |  |
| densenet+convnext | densenet 0.255 / convnext 0.745 | 0.9080 | 0.7429 | 0.9130 | 0.8578 | 0.45 | 0.7810 | 0.9016 | 0.8624 | 0.6826 | 0.7923 | 0.7866 | TN 394 / FP 43 / FN 46 / TP 164 |  |
| densenet+swin | densenet 0.801 / swin 0.199 | 0.8988 | 0.5810 | 0.9703 | 0.8439 | 0.27 | 0.7905 | 0.8970 | 0.8624 | 0.6875 | 0.7867 | 0.7886 | TN 392 / FP 45 / FN 44 / TP 166 |  |
| effnet+convnext | effnet 0.760 / convnext 0.240 | 0.9094 | 0.6952 | 0.9405 | 0.8609 | 0.35 | 0.8048 | 0.8787 | 0.8547 | 0.6835 | 0.7613 | 0.7824 | TN 384 / FP 53 / FN 41 / TP 169 |  |
| effnet+densenet | effnet 0.713 / densenet 0.287 | 0.9047 | 0.6000 | 0.9451 | 0.8331 | 0.29 | 0.8095 | 0.8787 | 0.8563 | 0.6882 | 0.7623 | 0.7852 | TN 384 / FP 53 / FN 40 / TP 170 |  |
| effnet+swin | effnet 0.700 / swin 0.300 | 0.9070 | 0.7048 | 0.9336 | 0.8594 | 0.35 | 0.8190 | 0.8879 | 0.8655 | 0.7069 | 0.7783 | 0.7981 | TN 388 / FP 49 / FN 38 / TP 172 |  |
| densenet+convnext+swin | densenet 0.799 / convnext 0.002 / swin 0.199 | 0.8989 | 0.5810 | 0.9703 | 0.8439 | 0.27 | 0.7905 | 0.8970 | 0.8624 | 0.6875 | 0.7867 | 0.7886 | TN 392 / FP 45 / FN 44 / TP 166 |  |
| effnet+convnext+swin | effnet 0.688 / convnext 0.075 / swin 0.237 | 0.9085 | 0.7095 | 0.9359 | 0.8624 | 0.35 | 0.8286 | 0.8833 | 0.8655 | 0.7119 | 0.7733 | 0.8000 | TN 386 / FP 51 / FN 36 / TP 174 |  |
| effnet+densenet+convnext | effnet 0.623 / densenet 0.291 / convnext 0.086 | 0.9087 | 0.6286 | 0.9451 | 0.8423 | 0.31 | 0.8048 | 0.8947 | 0.8655 | 0.6995 | 0.7860 | 0.7953 | TN 391 / FP 46 / FN 41 / TP 169 |  |
| effnet+densenet+swin | effnet 0.640 / densenet 0.084 / swin 0.276 | 0.9078 | 0.7048 | 0.9451 | 0.8671 | 0.34 | 0.8238 | 0.8856 | 0.8655 | 0.7094 | 0.7758 | 0.7991 | TN 387 / FP 50 / FN 37 / TP 173 |  |
| effnet+densenet+convnext+swin | effnet 0.673 / densenet 0.019 / convnext 0.074 / swin 0.234 | 0.9088 | 0.7095 | 0.9405 | 0.8655 | 0.35 | 0.8286 | 0.8833 | 0.8655 | 0.7119 | 0.7733 | 0.8000 | TN 386 / FP 51 / FN 36 / TP 174 |  |

## Best Schemes by Category

| Combination | Weight | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | Optimal Threshold | Optimal Sens | Optimal Spec | Optimal Acc | Youden J | Precision | F1-Score | Confusion Matrix | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| effnet+convnext | effnet 0.427 / convnext 0.573 | 0.9133 | 0.7381 | 0.9153 | 0.8578 | 0.42 | 0.7857 | 0.8902 | 0.8563 | 0.6759 | 0.7746 | 0.7801 | TN 389 / FP 48 / FN 45 / TP 165 | Two-model best AUC |
| effnet+swin | effnet 0.700 / swin 0.300 | 0.9070 | 0.7048 | 0.9336 | 0.8594 | 0.35 | 0.8190 | 0.8879 | 0.8655 | 0.7069 | 0.7783 | 0.7981 | TN 388 / FP 49 / FN 38 / TP 172 | Two-model best Youden |
| convnext+swin | convnext 0.602 / swin 0.398 | 0.9067 | 0.7619 | 0.8947 | 0.8516 | 0.64 | 0.7381 | 0.9336 | 0.8702 | 0.6717 | 0.8424 | 0.7868 | TN 408 / FP 29 / FN 55 / TP 155 | Two-model best Accuracy |
| effnet+densenet+convnext | effnet 0.358 / densenet 0.244 / convnext 0.398 | 0.9138 | 0.7095 | 0.9382 | 0.8640 | 0.32 | 0.8095 | 0.8627 | 0.8454 | 0.6722 | 0.7391 | 0.7727 | TN 377 / FP 60 / FN 40 / TP 170 | Three-model best AUC |
| effnet+convnext+swin | effnet 0.688 / convnext 0.075 / swin 0.237 | 0.9085 | 0.7095 | 0.9359 | 0.8624 | 0.35 | 0.8286 | 0.8833 | 0.8655 | 0.7119 | 0.7733 | 0.8000 | TN 386 / FP 51 / FN 36 / TP 174 | Three-model best Youden |
| effnet+convnext+swin | effnet 0.142 / convnext 0.565 / swin 0.293 | 0.9104 | 0.7619 | 0.9085 | 0.8609 | 0.63 | 0.7333 | 0.9405 | 0.8733 | 0.6738 | 0.8556 | 0.7897 | TN 411 / FP 26 / FN 56 / TP 154 | Three-model best Accuracy |
| effnet+densenet+convnext+swin | effnet 0.358 / densenet 0.244 / convnext 0.398 / swin 0.000 | 0.9138 | 0.7095 | 0.9382 | 0.8640 | 0.32 | 0.8095 | 0.8627 | 0.8454 | 0.6722 | 0.7391 | 0.7727 | TN 377 / FP 60 / FN 40 / TP 170 | Four-model best AUC; Swin weight is 0, effectively degrades to three-model |
| effnet+densenet+convnext+swin | effnet 0.364 / densenet 0.234 / convnext 0.382 / swin 0.020 | 0.9138 | 0.7095 | 0.9382 | 0.8640 | 0.33 | 0.8048 | 0.8673 | 0.8470 | 0.6720 | 0.7445 | 0.7735 | TN 379 / FP 58 / FN 41 / TP 169 | Four-model AUC non-zero weight candidate |
| effnet+densenet+convnext+swin | effnet 0.673 / densenet 0.019 / convnext 0.074 / swin 0.234 | 0.9088 | 0.7095 | 0.9405 | 0.8655 | 0.35 | 0.8286 | 0.8833 | 0.8655 | 0.7119 | 0.7733 | 0.8000 | TN 386 / FP 51 / FN 36 / TP 174 | Four-model best Youden |
| effnet+densenet+convnext+swin | effnet 0.140 / densenet 0.046 / convnext 0.553 / swin 0.261 | 0.9109 | 0.7571 | 0.9108 | 0.8609 | 0.62 | 0.7333 | 0.9405 | 0.8733 | 0.6738 | 0.8556 | 0.7897 | TN 411 / FP 26 / FN 56 / TP 154 | Four-model best Accuracy |

## Recommended Candidates

| Combination | Weight | AUC | Sens @0.50 | Spec @0.50 | Acc @0.50 | Optimal Threshold | Optimal Sens | Optimal Spec | Optimal Acc | Youden J | Precision | F1-Score | Confusion Matrix | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| effnet+densenet+convnext | effnet 0.358 / densenet 0.244 / convnext 0.398 | 0.9138 | 0.7095 | 0.9382 | 0.8640 | 0.32 | 0.8095 | 0.8627 | 0.8454 | 0.6722 | 0.7391 | 0.7727 | TN 377 / FP 60 / FN 40 / TP 170 | AUC-priority: current highest ranking ability; Swin weight was driven to 0 by search, indicating it does not further improve AUC. |
| effnet+densenet+convnext+swin | effnet 0.673 / densenet 0.019 / convnext 0.074 / swin 0.234 | 0.9088 | 0.7095 | 0.9405 | 0.8655 | 0.35 | 0.8286 | 0.8833 | 0.8655 | 0.7119 | 0.7733 | 0.8000 | TN 386 / FP 51 / FN 36 / TP 174 | Balanced screening priority: highest Youden J, with Sensitivity reaching 0.8286 and Accuracy 0.8655. |
| effnet+densenet+convnext+swin | effnet 0.364 / densenet 0.234 / convnext 0.382 / swin 0.020 | 0.9138 | 0.7095 | 0.9382 | 0.8640 | 0.33 | 0.8048 | 0.8673 | 0.8470 | 0.6720 | 0.7445 | 0.7735 | TN 379 / FP 58 / FN 41 / TP 169 | Four-model non-zero candidate: all four models have non-zero weights, AUC nearly equals the highest point, suitable when four-model fusion needs to be demonstrated. |
| effnet+convnext | effnet 0.427 / convnext 0.573 | 0.9133 | 0.7381 | 0.9153 | 0.8578 | 0.42 | 0.7857 | 0.8902 | 0.8563 | 0.6759 | 0.7746 | 0.7801 | TN 389 / FP 48 / FN 45 / TP 165 | Lightweight two-model candidate: EffNet+ConvNeXt alone approaches the highest AUC, with lower deployment complexity than three/four models. |
| effnet+densenet+convnext+swin | effnet 0.140 / densenet 0.046 / convnext 0.553 / swin 0.261 | 0.9109 | 0.7571 | 0.9108 | 0.8609 | 0.62 | 0.7333 | 0.9405 | 0.8733 | 0.6738 | 0.8556 | 0.7897 | TN 411 / FP 26 / FN 56 / TP 154 | Accuracy operating point candidate: highest Accuracy, but recall is relatively low, suitable as a conservative operating point reference. |

## Conclusion

- If only looking at AUC, the optimal scheme remains `effnet+densenet+convnext` with weights `0.358 / 0.244 / 0.398` and AUC `0.9138`.
- Adding Swin does not improve the highest AUC; the four-model AUC optimal point drives Swin weight to `0.000`, effectively equivalent to the three-model ensemble.
- If screening operating points are prioritized, the four-model scheme is more valuable: `effnet 0.673 / densenet 0.019 / convnext 0.074 / swin 0.234`, Youden J `0.7119`, Sensitivity `0.8286`, Specificity `0.8833`.
- If deployment complexity is a concern, the `effnet+convnext` two-model AUC of `0.9133` is very close to the three-model highest AUC and is the most worth retaining as a lightweight candidate.
