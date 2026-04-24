# BUCAD

[中文说明](README_CN.md)

BUCAD is a Windows-first breast ultrasound computer-aided diagnosis prototype.
It provides an end-to-end workflow for dataset preparation, classifier training,
external BUSI evaluation, lesion visualization, Grad-CAM-style explanation,
Gradio demo delivery, release packaging, and report/defense handoff.

> This project is for research, competition demonstration, and auxiliary
> analysis only. It must not be used as a clinical diagnosis replacement.

## Current Status

- Feature tasks: 79/79 complete in `specs/001-breast-ultrasound-cad/tasks.md`.
- Runtime classifier: five-fold `tf_efficientnetv2_s` ensemble.
- Runtime segmenter: `artifacts/checkpoints/segmenter_fold1.pt`.
- Runtime threshold: `0.25`, selected from BUSI threshold analysis.
- Latest verification: `26 passed` through `check_all.py` in the `BUCAD` Conda environment.
- Formal comparison: `tf_efficientnetv2_s` has the highest completed fold-1 AUC among the recorded baseline comparison run.

## Final Model Evidence

### BUSI External Evaluation

The frozen EfficientNetV2-S ensemble reaches the following BUSI external
evaluation results at the selected high-sensitivity operating point:

| Metric | Value |
| --- | ---: |
| AUC | 0.8955 |
| Sensitivity | 0.8476 |
| Specificity | 0.8215 |
| Accuracy | 0.8300 |
| Selected threshold | 0.25 |

The AUC target (`>= 0.75`) is exceeded. The sensitivity target (`0.85`) is
nearly reached, short by `0.0024`.

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

`alexnet` support has since been added through `torchvision`; rerun T060 if an
updated AlexNet metric is required in the final report.

## Repository Layout

- `configs/`: YAML configuration files for paths, classifiers, segmenter, and inference.
- `src/datasets/`: BUSBRA and BUSI dataset loading plus split support.
- `src/models/`: classifier and segmenter factories.
- `src/engine/`: training, comparison, inference, and error-handling workflows.
- `src/explain/`: Grad-CAM and overlay generation.
- `src/preprocess/`: image I/O and preprocessing transforms.
- `src/utils/`: config, metrics, reports, paths, logging, and result schemas.
- `scripts/`: command-line entry points for training, evaluation, export, and packaging.
- `app/`: Gradio application.
- `tests/`: unit and smoke tests.
- `specs/001-breast-ultrasound-cad/`: Spec Kit plan, spec, contracts, quickstart, and task list.
- `artifacts/`: local runtime outputs, checkpoints, reports, and release bundles.

## Data Governance

- BUSBRA is used for training and internal validation.
- BUSI is reserved for external evaluation and demo validation.
- Split generation is case-level to avoid leakage.
- Datasets, checkpoints, generated images, JSON/CSV reports, and release bundles
  are ignored or kept local by default.

Expected local dataset paths are configured in `configs/paths.local.yml`:

```yaml
datasets:
  busbra_root: ./训练集/BUSBRA
  busi_root: ./测试集/Dataset_BUSI_with_GT
```

## Environment Setup

Recommended Windows PowerShell workflow:

```powershell
conda create -n BUCAD python=3.11 -y
conda activate BUCAD
python -m pip install -r requirements.txt
```

If you are running commands from an external agent or non-activated shell, use:

```powershell
conda run -n BUCAD python check_env.py
```

## Verify The Project

```powershell
python check_env.py
python check_all.py
```

`check_all.py` verifies dependencies and runs the unit/smoke suite with a
repository-local temporary directory to avoid Windows Temp permission issues.

## Prepare Data Splits

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

Expected outputs:

- `artifacts/reports/busbra_5fold_splits.csv`
- `artifacts/reports/busbra_split_summary.json`

## Train Classifiers

Train the current handbook main model for one fold:

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

Train all five EfficientNetV2-S folds:

```powershell
scripts\train_all_folds.bat
```

Equivalent explicit commands:

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 2
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 3
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 4
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 5
```

Expected outputs:

- `artifacts/checkpoints/efficientnetv2_s_fold1.pt` through `efficientnetv2_s_fold5.pt`
- `artifacts/reports/train_cls_efficientnetv2_s_fold1.json` through `fold5.json`

## Run Model Comparison

Fast configuration check:

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --model-limit 1 --dry-run
```

Full handbook comparison:

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
```

Expected outputs:

- `artifacts/reports/comparison_results.json`
- `artifacts/reports/comparison_summary.md`
- per-model checkpoints under `artifacts/checkpoints/`
- per-model reports under `artifacts/reports/comparison_*_fold1.json`

## Run BUSI External Evaluation

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_eval_final.json
```

Expected outputs:

- `artifacts/reports/busi_eval_final.json`
- `artifacts/reports/threshold_analysis.md`

Note: the evaluator keeps conventional `0.50` metrics under the top-level
`metrics` key and writes the selected operating point under
`threshold_analysis.best_by_youden`.

## Launch The Gradio Demo

```powershell
python app\main.py
```

The UI accepts one breast ultrasound image and returns:

- benign/malignant probabilities
- final judgment and confidence/borderline information
- lesion localization overlay
- Grad-CAM-style explanation heatmap
- warnings when optional outputs are missing

## Export Evidence And Reports

Visual evidence:

```powershell
python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence_final --limit 6
```

Batch inference:

```powershell
python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference_final.csv
```

Editable Word report:

```powershell
python scripts\export_report_documents.py --reports-dir artifacts\reports --output-dir artifacts\reports\documents
```

Important report files:

- `artifacts/reports/efficientnetv2_s_5fold_summary.md`
- `artifacts/reports/model_freeze_decision.md`
- `artifacts/reports/comparison_summary.md`
- `artifacts/reports/report_tables.md`
- `artifacts/reports/defense_outline.md`
- `artifacts/reports/defense_qa.md`
- `artifacts/reports/final_handoff.md`
- `artifacts/reports/documents/bucad_report_summary.docx`

## Build A Release Bundle

```powershell
pyinstaller packaging\demo.spec --noconfirm
python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
```

Expected outputs:

- `dist/bucad-demo/bucad-demo.exe`
- `artifacts/release_v1/`
- `artifacts/reports/release_v1_manifest.md`
- `artifacts/reports/final_packaged_demo.md`

## Recommended Final Workflow

```powershell
conda activate BUCAD
python check_all.py
python scripts\make_split.py --config configs\paths.local.yml
scripts\train_all_folds.bat
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_eval_final.json
python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence_final --limit 6
python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference_final.csv
python scripts\export_report_documents.py --reports-dir artifacts\reports --output-dir artifacts\reports\documents
python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
```

## Limitations

- The current system is a prototype and should be described as auxiliary analysis only.
- Sensitivity is very close to the handbook target but not strictly above `0.85`.
- Grad-CAM and segmentation outputs are explanatory evidence, not clinical ground truth.
- Final presentation slides should still be manually reviewed for visual plausibility and wording.
