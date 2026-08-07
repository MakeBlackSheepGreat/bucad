# LesioNeXt Failed-Candidate Archive

This folder is a read-only snapshot of LesioNeXt candidates that failed the
current promotion protocol or were superseded by the frozen LENS v1a paper
mainline. The original checkpoint and configuration files remain in their
working locations for reproducibility; active paper evaluation must use only
the v1a five-fold evidence-only configuration.

## Frozen Mainline

- Model: `LesioNeXt-LENS v1a`
- Checkpoints: `artifacts/checkpoints/lesionext_lens_v1a_evidence_only_fold{1..5}.pt`
- Training config: `configs/classifier/lesionext_lens_v1a_evidence_only.yml`
- External inference config: `configs/inference/lesionext_lens_v1a_5fold_identity.yml`

## Archived Families

| Family | Reason for archive |
|---|---|
| LENS v1 | Five-fold OOF did not exceed the ConvNeXt-Tiny control |
| LENS v1a no-alignment control | Diagnostic ablation, not a promoted model |
| LENS v2 Confidence Gate | Accuracy and Sensitivity failed the fold1 gate |
| LENS v3 Label-Aware | Accuracy and Specificity fell below v1a; five-fold expansion stopped |
| LENS v1b Sensitivity Rank | AUC, Accuracy, Sensitivity, Specificity, and F1 all missed the v1a promotion gate |
| MoE V3 / V3.1 | Historical routing candidates, superseded as the paper mainline |
| MoE V4 / V4.1 / V4.2 / V4.3 | Teacher-constrained candidates without stable promotion evidence |
| AttnRes V4.4 / V4.5 / V4.6 | Fold1 performance remained below the matched ConvNeXt-Tiny control |
| Legacy unified LesioNeXt fold1 | Historical prototype checkpoint |

`archive_manifest.json` records the archived checkpoint, configuration, and
report counts. The archive is for audit and comparison only. It must not be
used to select new external-test settings.
