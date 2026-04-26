# Formal Experiment Commands

Date: 2026-04-24

## Execution Rules

- Run every command from the repository root: `C:\Users\876762330\Desktop\Agent`.
- Use the existing Conda environment through `conda run -n BUCAD ...`.
- Keep BUSBRA for training/internal validation only.
- Keep BUSI for external evaluation and demo validation only.
- Do not commit datasets, checkpoints, release bundles, or generated large artifacts.

## Preflight

```powershell
conda run -n BUCAD python check_env.py
conda run -n BUCAD python check_all.py
conda run -n BUCAD python scripts\make_split.py --config configs\paths.local.yml
```

Expected evidence:

- `check_env.py` reports required and optional packages available.
- `check_all.py` passes the smoke/unit suite.
- `artifacts/reports/busbra_5fold_splits.csv` exists and has `leakage_detected=false` in `artifacts/reports/busbra_split_summary.json`.

## T059: EfficientNetV2-S 5-Fold Training

```powershell
conda run -n BUCAD python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
conda run -n BUCAD python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 2
conda run -n BUCAD python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 3
conda run -n BUCAD python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 4
conda run -n BUCAD python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 5
```

Expected evidence:

- `artifacts/checkpoints/efficientnetv2_s_fold1.pt`
- `artifacts/checkpoints/efficientnetv2_s_fold2.pt`
- `artifacts/checkpoints/efficientnetv2_s_fold3.pt`
- `artifacts/checkpoints/efficientnetv2_s_fold4.pt`
- `artifacts/checkpoints/efficientnetv2_s_fold5.pt`
- `artifacts/reports/train_cls_efficientnetv2_s_fold1.json`
- `artifacts/reports/train_cls_efficientnetv2_s_fold2.json`
- `artifacts/reports/train_cls_efficientnetv2_s_fold3.json`
- `artifacts/reports/train_cls_efficientnetv2_s_fold4.json`
- `artifacts/reports/train_cls_efficientnetv2_s_fold5.json`

## T060: Full Model Comparison

```powershell
conda run -n BUCAD python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
```

Expected evidence:

- `artifacts/reports/comparison_results.json`
- Each comparison result has `status=completed` or a documented failure reason.

## T061: Final BUSI External Evaluation

After final classifier checkpoints are frozen, update `configs/inference/demo.yml`
to reference the selected classifier checkpoint or ensemble, then run:

```powershell
conda run -n BUCAD python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_eval_final.json
```

Expected evidence:

- `artifacts/reports/busi_eval_final.json`
- Metrics include AUC, sensitivity, specificity, accuracy, confusion matrix, and threshold analysis.

## T062: Freeze Decision

Create `artifacts/reports/model_freeze_decision.md` after T059-T061 are complete.
It must cite:

- Selected classifier checkpoint or ensemble.
- EfficientNetV2-S fold metrics.
- Full comparison results.
- BUSI external evaluation metrics.
- Final threshold and confidence/borderline handling.
- Known limitations and recovery plan.

## Follow-On Release Commands

```powershell
conda run -n BUCAD python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence_final --limit 6
conda run -n BUCAD python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference_final.csv
conda run -n BUCAD python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
conda run -n BUCAD python scripts\export_report_documents.py --reports-dir artifacts\reports --output-dir artifacts\reports\documents
```

Expected evidence:

- `artifacts/reports/visual_evidence_final/README.md`
- `artifacts/reports/batch_inference_final.csv`
- `artifacts/reports/release_v1_manifest.md`
- `artifacts/reports/documents/bucad_report_summary.docx`
