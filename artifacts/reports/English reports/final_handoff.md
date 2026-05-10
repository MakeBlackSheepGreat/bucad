# Final Handoff Checklist

Date: 2026-04-24

## Commands

- Environment check: `conda run -n BUCAD python check_env.py`
- Regression: `conda run -n BUCAD python check_all.py`
- BUSI final eval: `conda run -n BUCAD python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts
eportsusi_eval_final.json`
- Visual export: `conda run -n BUCAD python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts
eports
isual_evidence_final --limit 6`
- Batch export: `conda run -n BUCAD python scriptsatch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts
eportsatch_inference_final.csv`
- Release export: `conda run -n BUCAD python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts
elease_v1`

## Key Artifacts

- `artifacts/reports/busi_eval_final.json`
- `artifacts/reports/model_freeze_decision.md`
- `artifacts/reports/visual_evidence_final/README.md`
- `artifacts/reports/batch_inference_final.csv`
- `artifacts/reports/release_v1_manifest.md`
- `artifacts/reports/final_packaged_demo.md`
- `artifacts/reports/defense_outline.md`
- `artifacts/reports/defense_qa.md`

## Risks

- Formal EfficientNetV2-S 5-fold training remains pending.
- Full comparison run remains pending.
- BUSI sensitivity is below the handbook target, so avoid clinical-performance overclaiming.
- Final live screenshots still need manual capture for PPT polish.

## Suggested Owners

- Model owner: run T059/T060 long experiments and update freeze decision.
- Demo owner: rehearse UI and capture screenshots.
- Report owner: integrate tables, visual examples, limitations, and Q&A.
## Competition Metric Completeness Note

The official competition metric set is AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score. If an old archived table shows `-` for Precision or F1-Score, the historical summary did not preserve the confusion matrix or raw probabilities needed to reconstruct that value. For locked BUSI operating-point results, use `artifacts/reports/competition_metrics_all_busi_reports.md` as the complete metric source.
