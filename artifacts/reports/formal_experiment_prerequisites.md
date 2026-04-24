# Formal Experiment Prerequisites

Date: 2026-04-24

## Scope

This checklist verifies that the project is ready to run the handbook formal
experiments for final model evidence, including EfficientNetV2-S 5-fold training,
the full classifier comparison, BUSI external evaluation, and release freeze.

## Environment

| Item | Status | Evidence |
| --- | --- | --- |
| Conda environment | PASS | `conda run -n BUCAD python --version` reports Python 3.11.15 |
| Required Python packages | PASS | `conda run -n BUCAD python check_env.py` reports required and optional packages available |
| CUDA runtime | PASS | PyTorch reports CUDA available |
| GPU | PASS | NVIDIA GeForce RTX 5060 Laptop GPU |
| Regression baseline | PASS | `conda run -n BUCAD python check_all.py` passed 26 tests |

## Data Availability

| Dataset | Required Role | Local Path | Status | Evidence |
| --- | --- | --- | --- | --- |
| BUSBRA | Training and internal validation only | `训练集/BUSBRA` | PASS | 3,754 files detected under `训练集` |
| BUSI | External evaluation and demo validation only | `测试集/Dataset_BUSI_with_GT` | PASS | 1,578 files detected under `测试集` |
| BUSBRA 5-fold split | Case-level split assignments | `artifacts/reports/busbra_5fold_splits.csv` | PASS | 1,875 samples, 1,064 unique cases, leakage_detected=false |

## Current Checkpoints

| Checkpoint | Status | Notes |
| --- | --- | --- |
| `artifacts/checkpoints/classifier_fold1.pt` | AVAILABLE | Existing smoke/formal seed classifier checkpoint |
| `artifacts/checkpoints/segmenter_fold1.pt` | AVAILABLE | Existing segmentation checkpoint for visual-evidence flow |
| `artifacts/checkpoints/efficientnetv2_s_fold{fold}.pt` | PENDING | To be produced by T059 5-fold EfficientNetV2-S training |

## Current Reports

| Report | Status | Notes |
| --- | --- | --- |
| `artifacts/reports/train_cls_fold1.json` | AVAILABLE | Existing classifier training evidence |
| `artifacts/reports/busi_eval.json` | AVAILABLE | Existing BUSI evaluation evidence |
| `artifacts/reports/comparison_results.json` | PARTIAL | Current file is a dry-run/smoke comparison, not the full handbook comparison |
| `artifacts/reports/threshold_analysis.md` | AVAILABLE | Existing threshold-analysis note |
| `artifacts/reports/final_validation.md` | AVAILABLE | Existing regression/demo validation evidence |

## Formal Experiment Readiness

- PASS: Dataset boundary is ready: BUSBRA remains training/internal validation only, and BUSI remains external evaluation/demo validation only.
- PASS: Case-level split file exists and records no leakage.
- PASS: The BUCAD environment can import PyTorch, OpenCV, timm, segmentation_models_pytorch, and Gradio.
- PASS: CUDA is available for long-running formal training.
- PENDING: Final EfficientNetV2-S 5-fold checkpoints and reports must still be generated.
- PENDING: Full handbook comparison must replace the current dry-run comparison report.
- PENDING: Final BUSI evaluation should be rerun after the final classifier or ensemble is frozen.

## Next Commands

Run the commands recorded in `artifacts/reports/formal_experiment_commands.md`.
