<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Lesion-Area-Aware Dynamic Model Weight OOF Experiment

Date: 2026-04-26

## Objective

- Analyze the impact of lesion area size on EfficientNetV2-S and ConvNeXt-Tiny performance using BUSBRA OOF predictions.
- Dynamically adjust ROI branch model fusion weights based on lesion area bins.
- Parameter selection uses OOF only; external results serve as a single post-fixation review.

## Area-Bin Model Performance

| Area Bin | Samples | Positive | Negative | full Eff AUC | full Conv AUC | ROI Eff AUC | ROI Conv AUC | full Fixed AUC | ROI Fixed AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 258 | 32 | 226 | 0.7548 | 0.7933 | 0.5763 | 0.6861 | 0.7966 | 0.6874 |
| medium | 763 | 189 | 574 | 0.9007 | 0.9155 | 0.7722 | 0.8940 | 0.9306 | 0.8881 |
| large | 523 | 207 | 316 | 0.8817 | 0.8969 | 0.8243 | 0.8925 | 0.9095 | 0.8940 |
| very_large | 331 | 179 | 152 | 0.8905 | 0.8768 | 0.8584 | 0.8713 | 0.9040 | 0.8921 |

## OOF Selection Results

| Scheme | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current OOF ROI stack | 0.9239 | 0.550 | 0.8534 | 0.8423 | 0.8459 | 0.7214 | 0.7819 | TN 1068 / FP 200 / FN 89 / TP 518 |
| Area dynamic weight OOF candidate | 0.9232 | 0.550 | 0.8451 | 0.8699 | 0.8619 | 0.7566 | 0.7984 | TN 1103 / FP 165 / FN 94 / TP 513 |

OOF-selected ConvNeXt ROI weights:

- `small`: `0.00`
- `medium`: `0.40`
- `large`: `0.20`
- `very_large`: `0.60`

## Post-Fixation External Review

| Scheme | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current mainline | 0.9208 | 0.550 | 0.8524 | 0.8169 | 0.8284 | 0.6911 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |
| Area dynamic weight review | 0.9185 | 0.550 | 0.8333 | 0.8375 | 0.8362 | 0.7114 | 0.7675 | TN 366 / FP 71 / FN 35 / TP 175 |

## Conclusion

- Area dynamic weights did not exceed the current mainline; this direction is recommended for archival only.
- To merge into runtime, further implementation of per-ROI-area dynamic `classifier_weight_overrides` would be required.
