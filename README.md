# BUCAD

[中文说明](README_CN.md)

BUCAD is a Windows-first breast ultrasound computer-aided diagnosis prototype.
It covers the core engineering workflow for a breast ultrasound CAD project:
dataset preparation, leakage-safe splitting, classifier training, external
evaluation, lesion visualization, Grad-CAM-style explanation, Gradio demo
delivery, and release packaging.

> BUCAD is a research and prototype system. It is not a clinical diagnosis
> product and must not replace clinician judgment.

## What The Project Does

Given one breast ultrasound image, the system can:

- predict benign and malignant probabilities;
- apply a configurable decision threshold;
- return confidence and borderline warnings;
- generate lesion localization overlays;
- generate Grad-CAM-style explanation heatmaps;
- run BUSI folder-level batch inference;
- launch a Gradio demo for local review.

## Current Model

The current runtime classifier is a five-fold `tf_efficientnetv2_s` ensemble.
The inference config is stored in `configs/inference/demo.yml`.

Runtime settings:

- classifier: `tf_efficientnetv2_s`
- checkpoints: `artifacts/checkpoints/efficientnetv2_s_fold1.pt` through `efficientnetv2_s_fold5.pt`
- preprocessing: CLAHE enabled, horizontal-flip TTA enabled
- selected threshold: `0.25`
- segmenter checkpoint: `artifacts/checkpoints/segmenter_fold1.pt`

## Evaluation Summary

### BUSI External Evaluation

At the selected threshold (`0.25`), the current EfficientNetV2-S ensemble has:

| Metric | Value |
| --- | ---: |
| AUC | 0.8997 |
| Sensitivity | 0.8476 |
| Specificity | 0.8078 |
| Accuracy | 0.8207 |

The evaluator also keeps conventional `0.50` metrics in
`artifacts/reports/busi_tta_eval.json`. The selected operating point is stored
under `threshold_analysis.best_by_youden`.

### Model Comparison

Recorded fold-1, 20-epoch comparison:

| Rank | Model | AUC | Sensitivity | Specificity | Status |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `tf_efficientnetv2_s` | 0.8937 | 0.7049 | 0.8775 | completed |
| 2 | `densenet121` | 0.8867 | 0.7213 | 0.8775 | completed |
| 3 | `resnet18` | 0.8756 | 0.7131 | 0.8577 | completed |
| 4 | `mobilenetv3_small_100` | 0.8704 | 0.6557 | 0.9170 | completed |
| 5 | `basic_cnn` | 0.6427 | 0.0000 | 0.9921 | completed |
| 6 | `vgg16` | 0.5000 | 0.0000 | 1.0000 | completed |
| - | `alexnet` | - | - | - | failed in recorded run |

`alexnet` support was added after the recorded run. Rerun the comparison if an
updated AlexNet metric is required.

## Repository Layout

- `configs/`: YAML configuration files for paths, classifiers, segmenter, and inference.
- `src/datasets/`: BUSBRA and BUSI dataset loading plus split support.
- `src/models/`: classifier and segmenter factories.
- `src/engine/`: training, comparison, inference, and evaluation workflows.
- `src/explain/`: Grad-CAM and overlay generation.
- `src/preprocess/`: image I/O and preprocessing transforms.
- `src/utils/`: config, metrics, reports, paths, logging, and result schemas.
- `scripts/`: command-line entry points.
- `app/`: Gradio application.
- `tests/`: unit and smoke tests.
- `artifacts/`: local outputs, checkpoints, reports, and release bundles.

## Data Layout

Local paths are configured in `configs/paths.local.yml`:

```yaml
datasets:
  busbra_root: ./训练集/BUSBRA
  busi_root: ./测试集/Dataset_BUSI_with_GT
```

Data governance:

- BUSBRA is used for training and internal validation.
- BUSI is used for external evaluation and demo validation.
- Splits are case-level to avoid leakage.
- Datasets, checkpoints, generated images, JSON/CSV reports, and release bundles
  are local artifacts and are ignored by Git.

## Setup

```powershell
conda create -n BUCAD python=3.11 -y
conda activate BUCAD
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python check_env.py
python check_all.py
```

## Generate Splits

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

Outputs:

- `artifacts/reports/busbra_5fold_splits.csv`
- `artifacts/reports/busbra_split_summary.json`

## Train The Classifier

Train one fold:

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

Train all five folds:

```powershell
scripts\train_all_folds.bat
```

Outputs:

- `artifacts/checkpoints/efficientnetv2_s_fold1.pt` through `efficientnetv2_s_fold5.pt`
- `artifacts/reports/train_cls_efficientnetv2_s_fold1.json` through `fold5.json`

## Run Model Comparison

Fast dry run:

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --model-limit 1 --dry-run
```

Full comparison:

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
```

Outputs:

- `artifacts/reports/comparison_results.json`
- `artifacts/reports/comparison_summary.md`
- per-model checkpoints under `artifacts/checkpoints/`
- per-model reports under `artifacts/reports/comparison_*_fold1.json`

## Run BUSI Evaluation

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_tta_eval.json
```

Outputs:

- `artifacts/reports/busi_tta_eval.json`
- `artifacts/reports/threshold_analysis.md`

## Launch The Demo

```powershell
python app\main.py
```

The demo opens a local Gradio app for single-image diagnosis and visualization.

## Export Project Artifacts

Visual evidence:

```powershell
python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence_final --limit 6
```

Batch inference:

```powershell
python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference_final.csv
```

Word report summary:

```powershell
python scripts\export_report_documents.py --reports-dir artifacts\reports --output-dir artifacts\reports\documents
```

Release bundle:

```powershell
pyinstaller packaging\demo.spec --noconfirm
python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
```

## Useful Reports

- `artifacts/reports/efficientnetv2_s_5fold_summary.md`
- `artifacts/reports/model_freeze_decision.md`
- `artifacts/reports/comparison_summary.md`
- `artifacts/reports/report_tables.md`
- `artifacts/reports/resolution_augmentation_experiment.md`
- `artifacts/reports/final_validation.md`
