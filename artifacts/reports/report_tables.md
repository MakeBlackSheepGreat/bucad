# Report Tables

Status: BUSI threshold table populated from the current validated checkpoint.
Full handbook comparison still requires the long GPU run.

## Model Comparison Table

The current smoke evidence validates the comparison pipeline with `--dry-run`.
Populate metric columns after the full handbook comparison run.

| Model | Fold | AUC | Sensitivity | Specificity | Runtime Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic CNN | 1 | pending | pending | pending | 0.003 dry-run |

## BUSI Evaluation Table

Generated from `artifacts/reports/busi_eval.json` and
`artifacts/reports/threshold_analysis.md`.

| Threshold | AUC | Sensitivity | Specificity | Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.7564 | 0.6095 | 0.8009 | 0.7388 |
