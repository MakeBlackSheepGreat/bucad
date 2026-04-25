# Swin-Tiny Transformer Experiment

Date: 2026-04-25

## Goal

Evaluate whether a lightweight Transformer backbone can improve the current
breast ultrasound classifier baseline before spending time on five-fold
training or ensemble integration.

## Setup

| Item | Setting |
| --- | --- |
| Model | `swin_tiny_patch4_window7_224` |
| Training data | BUSBRA only |
| Fold | 1 |
| Epochs | 20 |
| Input size | 224 |
| Preprocessing | CLAHE |
| Augmentation | Horizontal flip |
| Learning rate | 0.0001 |
| BUSI role | External evaluation only; not used for training |

## Recipe Limitation

This was a quick project-recipe screening run, not a full Swin-specific
optimization. The run reused the current classifier pipeline: square resize,
CLAHE, horizontal flip, AdamW, fixed learning rate, no scheduler/warmup, and
last-checkpoint saving. It did not apply timm's full pretrained transform and
training recipe for Swin, such as ImageNet mean/std normalization, crop policy,
or Swin-specific fine-tuning choices.

Therefore this experiment can reject this exact quick recipe, but it should not
be used to claim that Swin Transformer is inherently unsuitable for breast
ultrasound classification.

## BUSBRA Fold-1 Validation

This table is a fold-1 single-model comparison. It should be used to judge the
backbone itself under a comparable validation split, not to compare against the
final optimized ensemble.

| Model | AUC | Sensitivity | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| EfficientNetV2-S fold1 comparison run | 0.8937 | 0.7049 | 0.8775 | 0.8213 |
| DenseNet121 fold1 | 0.8867 | 0.7213 | 0.8775 | 0.8267 |
| Swin-Tiny fold1 | 0.7950 | 0.8852 | 0.5020 | 0.6267 |

Swin-Tiny strongly increases sensitivity at the default threshold, but the gain
comes from predicting many more samples as malignant. Specificity and accuracy
drop sharply, and AUC is far below the fold-1 EfficientNetV2-S and DenseNet121
single-model baselines.

## BUSI External Evaluation

| Operating Point | AUC | Threshold | Sensitivity | Specificity | Accuracy | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default threshold | 0.8223 | 0.50 | 0.8810 | 0.5584 | 0.6631 | TN 244 / FP 193 / FN 25 / TP 185 |
| Best Youden point | 0.8223 | 0.57 | 0.7619 | 0.7483 | 0.7527 | TN 327 / FP 110 / FN 50 / TP 160 |

BUSI is used here only as an external sanity check for this single fold. The
default threshold gives high sensitivity, but it produces too many false
positives. This is not sufficient evidence to spend five-fold training time on
this exact Swin-Tiny setup.

## Decision

- Do not spend five-fold training time on this exact Swin-Tiny configuration
  based on the fold-1 single-model result.
- If Transformer-style models are still desired, try a hybrid or modern CNN
  candidate next, such as `convnext_tiny`, `maxvit_tiny_rw_224`, or
  `coatnet_0_rw_224`.
- For sensitivity improvement, the current evidence still favors threshold
  tuning, balanced-sensitive EfficientNet training, or heterogeneous ensemble
  refinement over this Swin-Tiny configuration.

## Artifacts

- Config: `configs/classifier/swin_tiny_patch4_window7_224.yml`
- Training report: `artifacts/reports/train_cls_swin_tiny_patch4_window7_224_fold1.json`
- Checkpoint: `artifacts/checkpoints/swin_tiny_patch4_window7_224_fold1.pt`
- BUSI report: `artifacts/reports/busi_swin_tiny_patch4_window7_224_fold1.json`
- BUSI threshold analysis: `artifacts/reports/threshold_analysis_swin_tiny_patch4_window7_224_fold1.md`
