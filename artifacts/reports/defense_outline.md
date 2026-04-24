# Defense Outline

Date: 2026-04-24

## 1. Problem And Goal

- Task: benign/malignant diagnosis assistance for breast ultrasound images.
- Deliverable: a runnable CAD prototype with classification, lesion localization, explanation heatmap, and Gradio demo.
- Boundary: auxiliary analysis only, not a clinical diagnosis replacement.

## 2. Data Governance

- BUSBRA is used for training and internal validation.
- BUSI is reserved for external evaluation and demo validation.
- Case-level split avoids leakage across folds.

## 3. Method

- Classifier pipeline: preprocessing, model inference, probability output, threshold-based decision, confidence/borderline handling.
- Localization pipeline: segmentation overlay for lesion-area visualization.
- Explainability pipeline: Grad-CAM-style heatmap for model attention review.
- Demo pipeline: Gradio upload, result panels, warning/fallback messages, release bundle.

## 4. Evidence

- Regression suite passes in the BUCAD environment.
- BUSI final evaluation: see `artifacts/reports/busi_eval_final.json`.
- Visual evidence: see `artifacts/reports/visual_evidence_final/README.md`.
- Release manifest: see `artifacts/reports/release_v1_manifest.md`.

## 5. Limitations And Next Work

- Current frozen model is a release-candidate prototype; formal EfficientNetV2-S 5-fold evidence is still pending.
- BUSI sensitivity is below the target for clinical screening use.
- Future work: complete full comparison, improve sensitivity, add clinician visual review, and calibrate threshold.
