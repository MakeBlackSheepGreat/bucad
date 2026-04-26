# BUCAD

[Chinese README](README_CN.md)

BUCAD (Breast Ultrasound Computer-Aided Diagnosis) is a Windows-first computer-aided diagnosis prototype for breast ultrasound imaging. It focuses on benign/malignant tumor classification, lesion-region assistance, and explainable visual output, combining classifier models, a lesion segmenter, ROI-guided inference, Grad-CAM heatmaps, a local Gradio demo, and Windows desktop packaging in one Python project.

The project is designed for medical-imaging algorithm validation, teaching demos, research prototyping, and secondary development. Third-party users can run the demo for single-image inference, or extend the existing training, evaluation, and reporting scripts for new model families, datasets, or inference strategies.

> BUCAD is a research prototype. It is not a clinical diagnosis product and must not replace clinician judgment.

## What The Project Does

Given one breast ultrasound image, the system can:

- predict benign and malignant probabilities;
- apply a configurable malignant decision threshold;
- return confidence and borderline warnings;
- generate lesion localization overlays when segmentation weights are available;
- generate Grad-CAM-style heatmaps from the main classifier;
- evaluate labeled dataset folders with reproducible metrics;
- launch a local Gradio web UI for single-image review.

Typical inference workflow:

1. Load the input image, apply grayscale handling, CLAHE enhancement, resizing, and model-specific normalization.
2. Run five-fold classifiers on the full image to estimate malignant probability.
3. Use the segmenter to predict a lesion mask and crop the ROI region.
4. Run the same classifier branches on the ROI crop for lesion-focused probability.
5. Fuse full-image and ROI probabilities with a lightweight OOF-trained logistic stacker.
6. Return the final class, confidence, lesion overlay, and Grad-CAM heatmap at the default threshold.

## Current Demo Model

The current mainline is `ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate`, configured in `configs/inference/demo.yml`.

- **Primary branch**: `ConvNeXt-Tiny`, five-fold checkpoints, ensemble weight `0.573`, timm-aware preprocessing, and crop-sweep TTA. It is also the declared main model for the web UI and Grad-CAM explanations.
- **Auxiliary branch**: `EfficientNetV2-S`, five-fold checkpoints, ensemble weight `0.427`, CLAHE preprocessing, and identity TTA. It complements ConvNeXt-Tiny while keeping deployment lighter than adding DenseNet/Swin.
- **ROI branch**: `segmenter_fold1.pt` predicts a lesion mask; training-set segmentation validation selects `mask_threshold=0.40`, `margin_ratio=0.35`, and largest-connected-component ROI cropping.
- **OOF fusion**: the system predicts both full-image probability and ROI probability, then combines them with a logit logistic stacker trained from training-set out-of-fold predictions.
- **ROI area gate**: ROI crops with area ratio below `0.08` or above `0.75` fall back to full-image prediction to reduce unstable ROI influence.
- **Decision threshold**: `0.510`, configured as the default operating point for the packaged demo.

Representative evaluation result for the demo model:

| Model | Threshold | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF Stacking + LCC ROI + Area Gate | 0.510 | 0.9256 | 0.8532 | 0.8667 | 0.7309 | 0.8467 | 0.7930 |

Optional retained configurations:

| Model | Threshold | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | Use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny + EfficientNetV2-S + DenseNet121 non-ROI optimized ensemble | 0.453 | 0.9162 | 0.8702 | 0.7476 | 0.8351 | 0.9291 | 0.7889 | optional high-specificity comparison |
| EfficientNetV2-S + DenseNet121 + ConvNeXt-Tiny + ROI OOF LCC | 0.560 | 0.9229 | 0.8501 | 0.8143 | 0.7467 | 0.8673 | 0.7790 | optional three-model ROI line |

These heavier three-model configurations are retained for reproducible experiments and side-by-side comparison. The default demo uses the two-model ROI Area Gate line because it reaches stronger AUC, Recall/Sensitivity, and F1-Score in the current evaluation while keeping deployment simpler than a 15-checkpoint three-model ensemble.
All model-result tables use AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score for consistent comparison.

## Model Design Rationale

The default configuration uses a two-model ROI-guided mainline rather than the heaviest available ensemble:

- **ConvNeXt-Tiny is the main model** because it is the strongest single-model family and gives Grad-CAM from the primary branch.
- **EfficientNetV2-S is retained** because it complements ConvNeXt and improves ensemble ranking with much lower complexity than adding DenseNet/Swin.
- **ROI guidance reduces background interference** by combining full-image context with lesion-focused local evidence.
- **Area gating improves robustness** by falling back to full-image prediction when the predicted ROI is too small or too large to be reliable.
- **DenseNet121 is kept in optional configurations**, but not used in the default demo because it adds five extra checkpoints while reducing the optimized demo operating-point sensitivity.
- **Swin-Tiny is not used in the default demo** because mixed-ensemble search showed limited contribution at the best operating point.

## Optimization Summary

Key inference optimizations:

1. **Heterogeneous preprocessing**: each ensemble member can define its own image size, CLAHE, normalization, interpolation, crop ratio, and TTA variants.
2. **ConvNeXt crop-sweep TTA**: ConvNeXt uses ImageNet mean/std, bicubic interpolation, and `crop_pct=0.90/0.95/1.00`; each crop evaluates identity and horizontal flip.
3. **EfficientNet identity branch**: EfficientNetV2-S uses CLAHE, area interpolation, no ConvNeXt mean/std inheritance, and identity TTA to avoid cross-model preprocessing mismatch.
4. **ROI OOF stacking**: the final demo combines full-image probability and ROI probability with an OOF-trained logit logistic stacker.
5. **ROI mask post-processing**: segmentation validation selects mask threshold `0.40`, largest connected component cropping, and `margin_ratio=0.35` before ROI inference.
6. **ROI area quality gate**: extremely small or overly large ROI crops fall back to full-image prediction, improving Precision/F1 at the current operating point.
7. **OOF threshold selection**: the final ROI Area Gate threshold is `0.510`, selected from training-set out-of-fold evidence.
8. **Demo alignment**: `demo.yml` places ConvNeXt-Tiny first, so the web UI and Grad-CAM explanation declare ConvNeXt-Tiny as the main classifier.

## Metric Definitions

Project reports and README benchmark tables include the following metrics:

- **AUC**: overall ability to rank benign and malignant samples.
- **Accuracy**: proportion of all correctly predicted samples.
- **Recall/Sensitivity**: proportion of malignant samples correctly identified as malignant.
- **Precision**: proportion of predicted malignant samples that are truly malignant.
- **Specificity**: proportion of benign samples correctly identified as benign.
- **F1-Score**: harmonic mean of Precision and Recall/Sensitivity.

## Repository Layout

- `configs/`: YAML configuration files for paths, classifiers, segmenter, and inference.
- `src/datasets/`: dataset loading and split support.
- `src/models/`: classifier and segmenter factories.
- `src/engine/`: training, comparison, inference, and evaluation workflows.
- `src/explain/`: Grad-CAM and overlay generation.
- `src/preprocess/`: image I/O and preprocessing transforms.
- `src/utils/`: config, metrics, reports, paths, logging, and result schemas.
- `scripts/`: command-line entry points.
- `app/`: Gradio web application.
- `tests/`: unit, integration, and smoke tests.
- `artifacts/reports/`: versioned Markdown experiment summaries.
- `artifacts/checkpoints/`: local model weights, ignored by Git.

## Data Layout

Create `configs/paths.local.yml` from `configs/paths.example.yml` and point it to local datasets:

```yaml
datasets:
  busbra_root: ./BUSBRA
  busi_root: ./Dataset_BUSI_with_GT
```

Dataset notes:

- Configure only the local dataset paths required for your workflow.
- Splits are case-level to avoid leakage.
- Datasets, checkpoints, generated images, JSON/CSV outputs, and release bundles are local artifacts and are ignored by Git.

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

## Hardware And Software Requirements

### Demo / Inference

- **Operating system**: Windows 10/11 x64 is the primary target. The source workflow also works on standard Python environments where PyTorch and OpenCV are available.
- **Packaged desktop demo**: No local Python installation is required after unpacking the release bundle. The desktop build uses Microsoft Edge WebView2; most Windows 10/11 machines already include it, otherwise install the official WebView2 Runtime.
- **CPU / GPU**: CPU inference is supported and is sufficient for ordinary demonstrations. NVIDIA GPU is optional and mainly improves inference latency.
- **Memory**: 8 GB RAM minimum, 16 GB RAM recommended for smoother startup and image visualization.
- **Disk space**: Reserve at least 8 GB for the unpacked demo, model checkpoints, temporary files, and generated visual outputs.

### Training / Experimentation

- **Python environment**: Conda with Python 3.10 or 3.11. The example setup uses a `BUCAD` environment with Python 3.11.
- **GPU**: NVIDIA CUDA GPU is strongly recommended. CPU-only training is suitable only for smoke tests, not for full five-fold model training.
- **VRAM**: 8 GB VRAM is a practical lower bound for the current 224-resolution ConvNeXt-Tiny / EfficientNetV2-S experiments. 12-16 GB or more is recommended for faster five-fold training, larger batch sizes, or higher-resolution trials.
- **System memory**: 16 GB RAM minimum, 32 GB RAM recommended for training, report generation, and parallel data loading.
- **Disk space**: Reserve at least 50 GB for local datasets, five-fold checkpoints, logs, OOF artifacts, reports, and temporary build outputs.
- **Dataset handling**: Keep dataset folders, checkpoints, generated reports, and release packages outside Git-tracked source files.

## Run The Demo

Browser mode:

```powershell
conda activate BUCAD
python app\main.py
```

Desktop window mode:

```powershell
conda activate BUCAD
python app\desktop_main.py
```

Both modes load `configs/inference/demo.yml`. The current demo declares `ConvNeXt-Tiny` as the primary model and uses `EfficientNetV2-S` as the auxiliary ensemble branch.

## One-Click Windows Demo

Browser-opening package:

```powershell
conda activate BUCAD
python -m PyInstaller --clean --noconfirm packaging\demo.spec
```

Desktop-window package:

```powershell
conda activate BUCAD
python -m PyInstaller --clean --noconfirm packaging\desktop_demo.spec
```

- Browser-opening executable: `dist/bucad-demo/bucad-demo.exe`.
- Desktop-window executable: `dist/bucad-demo-desktop/bucad-demo-desktop.exe`.
- Distribute and run the whole generated folder, not the single `.exe` file alone, because the executable depends on bundled model files, Python libraries, WebView files, and configs in the same directory.
- The `v1.1.0` release package uses the desktop-window build so the UI appears as a local Windows application instead of opening an external browser.

## Run Batch Evaluation

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_demo_convnext_effnet.json
```

Representative metrics for the packaged demo configuration:

| Model | Threshold | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate | 0.510 | 0.9256 | 0.8532 | 0.8667 | 0.7309 | 0.8467 | 0.7930 |

## Generate Splits

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

Outputs:

- `artifacts/reports/busbra_5fold_splits.csv`
- `artifacts/reports/busbra_split_summary.json`

## Train Classifiers

Train one fold:

```powershell
python scripts\train_cls.py --config configs\classifier\convnext_tiny_timm_recipe.yml --fold 1
```

Train EfficientNetV2-S one fold:

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

Five-fold checkpoints are expected under `artifacts/checkpoints/` and are intentionally not committed.

## Useful Reports

Additional experiment and result files:

- `artifacts/reports/fivefold_single_model_comparison.md`
- `artifacts/reports/fold1_single_model_baseline_comparison.md`
- `artifacts/reports/convnext_tta_optimization.md`
- `artifacts/reports/ensemble_tta_threshold_tuning.md`
- `artifacts/reports/roi_oof_experiment.md`
- `artifacts/reports/roi_oof_lcc_optimization.md`
- `artifacts/reports/roi_precision_f1_study.md`
- `artifacts/reports/oof_two_model_decision_report.md`
- `artifacts/reports/four_model_ensemble_weight_search.md`
- `artifacts/reports/three_model_ensemble_weight_search.md`
- `artifacts/reports/three_model_roi_oof_experiment.md`
- `artifacts/reports/swin_tiny_5fold_experiment.md`
- `artifacts/reports/报告/英文版报告/training_recipe_audit.md`
- `artifacts/reports/报告/英文版报告/literature_guided_optimization.md`
- `artifacts/reports/报告/英文版报告/final_validation.md`

Archived English reports are grouped under `artifacts/reports/报告/英文版报告/`, with synchronized Chinese versions under `artifacts/reports/报告/中文版报告/`.

## References And Acknowledgements

This project was inspired by public breast ultrasound datasets, open-source medical-imaging projects, and related research on classification, segmentation, ROI-aware diagnosis, multi-task learning, and ultrasound foundation models.

- BUSI dataset: [Dataset of breast ultrasound images](https://pubmed.ncbi.nlm.nih.gov/31867417/)
- Lesion-region-aware breast ultrasound classification: [PMC11431713](https://pmc.ncbi.nlm.nih.gov/articles/PMC11431713/)
- Multi-task breast ultrasound segmentation and classification: [PMC12011763](https://pmc.ncbi.nlm.nih.gov/articles/PMC12011763/)
- OpenUS ultrasound foundation model: [XZheng0427/OpenUS](https://github.com/XZheng0427/OpenUS)
- BUSI segmentation reference project: [tqxli/breast_ultrasound_lesion_segmentation_PyTorch](https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch)
- BUSI-SAM / SAM-style segmentation references: [huangjin520/BUSI-SAM](https://github.com/huangjin520/BUSI-SAM), [bscs12/BUSSAM](https://github.com/bscs12/BUSSAM)

We thank the authors and maintainers of these datasets, papers, and open-source projects. Their work provided valuable references for BUCAD's data handling, model comparison, segmentation visualization, mixed-ensemble design, and future ROI-aware optimization.
