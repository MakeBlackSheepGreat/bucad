# Final Validation Status

Date: 2026-04-23

## Environment

- Validation executed in Conda environment `BUCAD`
- Python `3.11.15`
- OS `Windows 10`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`

Note: the plan and quickstart were originally written for Python 3.10, but this final validation was completed on Python 3.11.15 per the current local environment decision on 2026-04-23.

## Quickstart Validation

The following commands were executed successfully in the current environment:

- `python check_env.py`
- `python check_all.py`
- `python scripts\make_split.py --config configs\paths.local.yml`
- `python scripts\train_cls.py --config configs\classifier\baseline.yml --fold 1 --epochs 1`
- `python scripts\train_seg.py --config configs\segmenter\unet.yml --fold 1 --epochs 1`
- `python scripts\eval_busi.py --config configs\inference\demo.yml`
- Source Gradio smoke: `build_app().launch(prevent_thread_lock=True, server_name='127.0.0.1', server_port=7861, share=False)` then closed cleanly
- `pyinstaller packaging\demo.spec --noconfirm`
- `python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir dist\bucad-demo`
- Packaged demo smoke: `dist\bucad-demo\bucad-demo.exe` stayed alive for 20 seconds with no stderr after exporting runtime configs and checkpoints into `dist\bucad-demo`

## Automated Regression Status

- Full test suite result after DOCX-only report export sync: `32 passed in 10.37s`

## Metrics And Outputs

- BUSBRA split generation:
  - samples: `1875`
  - unique cases: `1064`
  - folds: `5`
  - leakage detected: `false`
- Classifier fold 1 smoke training:
  - checkpoint: `artifacts/checkpoints/classifier_fold1.pt`
  - validation AUC: `0.8062`
  - validation sensitivity: `0.4098`
  - validation specificity: `0.9051`
  - validation accuracy: `0.7440`
- Segmenter fold 1 smoke training:
  - checkpoint: `artifacts/checkpoints/segmenter_fold1.pt`
  - validation Dice: `0.8100`
- BUSI external evaluation:
  - report: `artifacts/reports/busi_eval.json`
  - sample count: `647`
  - AUC: `0.7564`
  - sensitivity: `0.6095`
  - specificity: `0.8009`
  - accuracy: `0.7388`
- Single-image demo evidence:
  - input: `BUSI/malignant/malignant (1).png` from the local evaluation dataset root
  - report: `artifacts/reports/demo_single_image/diagnosis.json`
  - generated visuals:
    - `artifacts/reports/demo_single_image/original.png`
    - `artifacts/reports/demo_single_image/lesion_overlay.png`
    - `artifacts/reports/demo_single_image/explanation.png`

## Observations

- The end-to-end system now completes classification, lesion overlay generation, explanation overlay generation, Gradio launch, and packaged demo launch in the validated environment.
- The real single-image demo run produced a complete response with both visualization outputs, but the sampled malignant BUSI image `malignant (1).png` was classified as benign with malignant probability `0.3769`. This is acceptable for smoke validation, but it reinforces that the prototype is not clinically reliable and must remain an auxiliary demo system only.
- Packaging required runtime asset export into `dist\bucad-demo` so the executable could resolve `configs/inference/demo.yml` and the trained checkpoints from the packaged working directory. The quickstart and README were updated to match this validated flow.

## Handbook Follow-up Validation

Date: 2026-04-23

- Added handbook follow-up implementation paths for EfficientNetV2-S, comparison experiments, threshold analysis, visual evidence export, release manifests, and batch inference.
- Automated regression after handbook follow-up: `30 passed in 9.28s`.
- Handbook smoke commands completed:
  - `python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --model-limit 1 --dry-run`
  - `python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1`
  - `python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence --limit 1`
  - `python scripts\eval_busi.py --config configs\inference\demo.yml`
- BUSI threshold analysis generated `artifacts/reports/threshold_analysis.md` with best Youden threshold `0.50`.
- Full model-selection claims still require the long comparison run with `--epochs 20` or the final agreed epoch budget before report submission.

## Report Document Export Validation

Date: 2026-04-23

- Added readable Word DOCX report export through `scripts\export_report_documents.py`.
- Generated `artifacts\reports\documents\bucad_report_summary.docx` for editable report review.
- PDF export is intentionally out of scope after the 2026-04-23 delivery decision to use Word documents only.
- Automated validation completed with `python -m pytest tests\integration\test_report_documents.py`: `2 passed`.
- Full regression completed with `python -m pytest tests\unit tests\smoke tests\integration`: `32 passed in 10.37s`.
- Development handbook progress was synchronized in `artifacts\reports\handbook_progress.md` and included in the DOCX report summary.
