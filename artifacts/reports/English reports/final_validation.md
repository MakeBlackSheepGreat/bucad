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

## Model Pipeline Follow-up Validation

Date: 2026-04-23

- Added model pipeline follow-up implementation paths for EfficientNetV2-S, comparison experiments, threshold analysis, visual evidence export, release manifests, and batch inference.
- Automated regression after model pipeline follow-up: `30 passed in 9.28s`.
- Model pipeline smoke commands completed:
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
- 
## Release Candidate Follow-up Validation

Date: 2026-04-24

- Environment: `conda run -n BUCAD python check_all.py` completed with `26 passed in 9.29s`.
- Final visual evidence export completed: `artifacts/reports/visual_evidence_final/` with 6 cases.
- Final BUSI evaluation completed: `artifacts/reports/busi_eval_final.json` with AUC 0.7564, sensitivity 0.6095, specificity 0.8009, accuracy 0.7388.
- Final batch inference export completed: `artifacts/reports/batch_inference_final.csv`.
- Release asset export completed: `artifacts/release_v1/` and `artifacts/reports/release_v1_manifest.md`.
- Packaged demo smoke completed: `dist/bucad-demo/bucad-demo.exe` stayed alive for 20 seconds with empty stdout/stderr.
- Remaining long-running evidence: T059 EfficientNetV2-S 5-fold training and T060 full model comparison.

## EfficientNetV2-S T059 Completion

Date: 2026-04-24

- Completed all five EfficientNetV2-S training folds and generated checkpoints under `artifacts/checkpoints/efficientnetv2_s_fold{1..5}.pt`.
- Mean internal validation AUC: `0.8946`.
- Mean internal validation sensitivity: `0.6525`.
- Mean internal validation specificity: `0.9054`.
- Switched `configs/inference/demo.yml` to the EfficientNetV2-S five-fold ensemble.
- BUSI ensemble AUC: `0.8955`.
- Selected threshold by Youden J: `0.25` with sensitivity `0.8476`, specificity `0.8215`, and accuracy `0.8300`.
- Remaining formal task: T060 full model comparison.
## T060 Full Comparison Completion

Date: 2026-04-24

- Completed full model comparison command with 7 configured models, fold 1, 20 epochs.
- Best completed model by validation AUC: `tf_efficientnetv2_s` at `0.8937`.
- Comparison summary: `artifacts/reports/comparison_summary.md`.
- Full machine-readable result: `artifacts/reports/comparison_results.json`.
- Note: `alexnet` failed in this run and is documented as a failed baseline.


## Manual Recheck Fixes

Date: 2026-04-24

- Fixed comparison dry-run test isolation so check_all.py no longer overwrites formal rtifacts/reports/comparison_results.json.
- Added scripts/train_all_folds.bat to match the project quick command workflow.
- Added AlexNet construction support through torchvision so the configured comparison model is supported by the model factory.
- Updated check_all.py to use a unique repository-local pytest temp directory, avoiding locked Windows temp-directory failures.
- Verification: C:\Users\876762330\.conda\envs\BUCAD\python.exe check_all.py completed with 26 passed in 9.74s, and formal comparison remained dry_run=false, model_count=7.
## Competition Metric Completeness Note

The official competition metric set is AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score. If an old archived table shows `-` for Precision or F1-Score, the historical summary did not preserve the confusion matrix or raw probabilities needed to reconstruct that value. For locked BUSI operating-point results, use `artifacts/reports/competition_metrics_all_busi_reports.md` as the complete metric source.
