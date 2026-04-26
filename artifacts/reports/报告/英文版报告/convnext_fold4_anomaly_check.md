# ConvNeXt-Tiny Fold4 Anomaly Check

Date: 2026-04-25

## Question

ConvNeXt-Tiny timm-aware five-fold training shows fold4 AUC `0.8875`, which is
lower than folds 1, 2, 3, and 5. This report checks whether fold4 has a data,
checkpoint, preprocessing, or evaluation problem.

## Summary

No direct bug was found.

Fold4 is lower for ConvNeXt-Tiny, but the result is reproducible and the data
split looks balanced. The checkpoint saved for fold4 matches the best validation
AUC epoch, and independently reloading the checkpoint gives exactly the same
metrics as the training report.

## Fold4 Re-Evaluation

| Check | Result |
| --- | --- |
| Checkpoint | `artifacts/checkpoints/convnext_tiny_timm_recipe_fold4.pt` |
| Independent validation AUC | 0.8875 |
| Independent sensitivity | 0.7623 |
| Independent specificity | 0.8577 |
| Independent accuracy | 0.8267 |
| Confusion | TN 217 / FP 36 / FN 29 / TP 93 |
| Report match | Yes |

This rules out a simple reporting mismatch or checkpoint-loading mismatch.

## Training-Curve Check

| Fold | Best Epoch | Best AUC | Last Epoch AUC | Notes |
| ---: | ---: | ---: | ---: | --- |
| 1 | 13 | 0.9259 | 0.9243 | Stable high-AUC run |
| 2 | 3 | 0.9301 | 0.9137 | Early best, later lower AUC |
| 3 | 30 | 0.9469 | 0.9469 | Strongest run |
| 4 | 14 | 0.8875 | 0.8837 | Lower plateau, no sudden crash |
| 5 | 10 | 0.9158 | 0.9026 | Normal run |

Fold4 does not show a sudden collapse. It improves from epoch 1 AUC `0.8403`
to a best AUC of `0.8875`, then stays near `0.884` late in training. This
suggests a harder validation split or optimization variance rather than a
broken run.

## Split Distribution Check

| Fold | Train Images | Val Images | Val Benign | Val Malignant | Val Malignant Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1500 | 375 | 253 | 122 | 0.3253 |
| 2 | 1500 | 375 | 254 | 121 | 0.3227 |
| 3 | 1500 | 375 | 254 | 121 | 0.3227 |
| 4 | 1500 | 375 | 253 | 122 | 0.3253 |
| 5 | 1500 | 375 | 254 | 121 | 0.3227 |

The fold4 label distribution is almost identical to the other folds. No class
imbalance issue was found.

## Metadata Check

| Fold | BIRADS 2 | BIRADS 3 | BIRADS 4 | BIRADS 5 | View Side Pattern | Missing BBox |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 117 | 89 | 145 | 24 | balanced left/right/single | 0 |
| 2 | 103 | 93 | 136 | 43 | balanced left/right/single | 0 |
| 3 | 108 | 98 | 148 | 21 | balanced left/right/single | 0 |
| 4 | 113 | 90 | 142 | 30 | balanced left/right/single | 0 |
| 5 | 121 | 93 | 122 | 39 | balanced left/right/single | 0 |

Fold4 does not have an obvious metadata abnormality. BIRADS distribution and
view-side distribution are within the range of other folds.

## Cross-Model Context

| Fold | EfficientNetV2-S AUC | DenseNet121 AUC | ConvNeXt-Tiny AUC |
| ---: | ---: | ---: | ---: |
| 1 | 0.9248 | - | 0.9259 |
| 2 | 0.8988 | 0.8828 | 0.9301 |
| 3 | 0.9077 | 0.8849 | 0.9469 |
| 4 | 0.8891 | 0.8866 | 0.8875 |
| 5 | 0.8523 | 0.8334 | 0.9158 |

Fold4 is not uniquely bad for ConvNeXt-Tiny. EfficientNetV2-S and DenseNet121
also land around `0.886` to `0.889` on fold4. This strongly suggests that fold4
is a harder split for the current feature distribution, not a ConvNeXt-specific
pipeline error.

## Conclusion

Current evidence points to "harder fold / normal fold variance", not a data or
checkpoint bug.

Recommended next steps:

1. Keep fold4 in the five-fold ensemble; do not drop it without a formal rule.
2. If more certainty is needed, rerun only ConvNeXt fold4 with a different seed
   and compare whether AUC remains near `0.887` or returns above `0.91`.
3. If fold4 remains low across seeds, treat it as a harder validation subset and
   use it for error-case analysis rather than as a reason to discard ConvNeXt.
## Competition Metric Completeness Note

The official competition metric set is AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score. If an old archived table shows `-` for Precision or F1-Score, the historical summary did not preserve the confusion matrix or raw probabilities needed to reconstruct that value. For locked BUSI operating-point results, use `artifacts/reports/competition_metrics_all_busi_reports.md` as the complete metric source.
