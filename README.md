# BUCAD

BUCAD is a breast ultrasound computer-aided diagnosis prototype built around a
single Python codebase. The repository covers dataset preparation, model
training, external evaluation, explainability overlays, and a Gradio-based demo
interface.

## Quick Start

1. Create and activate the environment:

```powershell
conda create -n BUCAD python=3.10 -y
conda activate BUCAD
pip install -r requirements.txt
```

2. Verify the environment:

```powershell
python check_env.py
```

3. Copy the path template and point it at your local datasets:

```powershell
Copy-Item configs\paths.example.yml configs\paths.local.yml
```

4. Generate a leakage-safe split file:

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

5. Train and evaluate the baseline classifier:

```powershell
python scripts\train_cls.py --config configs\classifier\baseline.yml --fold 1 --epochs 1
python scripts\eval_busi.py --config configs\inference\demo.yml
```

To run the handbook main classifier configuration:

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

To generate model-selection evidence with a fast dry run first:

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --model-limit 1 --dry-run
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
```

6. Launch the demo:

```powershell
python app\main.py
```

7. Export visual evidence and batch predictions when preparing reports:

```powershell
python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence --limit 6
python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference.csv
```

8. Export readable DOCX/PDF report documents:

```powershell
python scripts\export_report_documents.py --reports-dir artifacts\reports --output-dir artifacts\reports\documents
```

The exporter summarizes the core Markdown and JSON reports into
`artifacts\reports\documents\bucad_report_summary.docx` and
`artifacts\reports\documents\bucad_report_summary.pdf` for review, archiving,
and defense preparation.

9. Build a distributable demo bundle:

```powershell
pyinstaller packaging\demo.spec --noconfirm
python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
```

## Repository Layout

- `configs/`: YAML configuration files.
- `src/`: Reusable Python modules for datasets, models, runtime, and inference.
- `scripts/`: CLI entry points for splits, training, evaluation, and export.
- `app/`: Gradio UI.
- `tests/`: Unit, integration, and smoke coverage.
- `artifacts/`: Runtime outputs, checkpoints, logs, and reports.

## Notes

- Training data and evaluation data are intentionally ignored by Git.
- The project is Windows-first and should still offer CPU-safe fallbacks where
  possible.
- Optional dependencies such as `timm`, `segmentation_models_pytorch`,
  `pytorch-grad-cam`, and `gradio` are used when installed and fail with clear
  messages otherwise.
