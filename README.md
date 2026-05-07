# BUCAD

[Chinese README](README_CN.md)

BUCAD (Breast Ultrasound Computer-Aided Diagnosis) is a Windows-first research prototype for breast ultrasound image analysis. It integrates benign/malignant classification, lesion-region guidance, explainable visualization, local Gradio inference, and Windows desktop packaging in one Python repository.

The project is intended for algorithm validation, reproducible benchmarking, teaching, and controlled secondary development. It is not a clinical product and must not be used as a substitute for physician judgment.

## Scope And Validation Boundary

- BUSBRA is used for training, internal validation, out-of-fold selection, and candidate screening.
- BUSI is used only after a candidate is frozen, for locked external validation and threshold confirmation.
- Project-level comparison tables report AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score.

## Validated Mainline

The deployed configuration is `configs/inference/demo.yml`.

- Primary classifier branch: `ConvNeXt-Tiny`, five-fold checkpoints, ensemble weight `0.573`, timm-aware preprocessing, and crop-sweep TTA.
- Auxiliary classifier branch: `EfficientNetV2-S`, five-fold checkpoints, ensemble weight `0.427`, CLAHE preprocessing, and identity TTA.
- ROI branch: `segmenter_fold1.pt` produces lesion masks; training-set validation selected `mask_threshold=0.40`, `margin_ratio=0.35`, and largest-connected-component ROI cropping.
- Fusion: full-image probability and ROI probability are combined by a logit-space logistic stacker trained on BUSBRA out-of-fold predictions.
- Quality gate: ROI crops with area ratio below `0.08` or above `0.75` fall back to the full-image branch.
- Operating threshold: `0.510`.

Validated BUSI external result:

| Model | Threshold | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF Stacking + LCC ROI + Area Gate | 0.510 | 0.9256 | 0.8532 | 0.8667 | 0.7309 | 0.8467 | 0.7930 |

Confusion matrix: TN 370 / FP 67 / FN 28 / TP 182.

## Representative Model Screening

The full screening archive is available in [`artifacts/reports/native_single_model_retest.md`](artifacts/reports/native_single_model_retest.md).

| Model | AUC | Recall/Sensitivity | Specificity | Note |
| --- | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny timm recipe fold1 | 0.8943 | 0.7762 | 0.8741 | recommended ConvNeXt recipe |
| ConvNeXt-Small timm recipe fold1 | 0.8947 | 0.7667 | 0.8581 | later upgrade reference |
| Swin-Tiny timm recipe fold1 | 0.8729 | 0.7048 | 0.8902 | Transformer-style candidate |
| DenseNet121 fold1 | 0.8766 | 0.4571 | 0.9771 | high-specificity reference |
| EfficientNetV2-S fold1 | 0.8609 | 0.8333 | 0.7048 | high-sensitivity CNN candidate |
| ResNet18 fold1 | 0.8480 | 0.7714 | 0.8124 | baseline reference |
| MobileNetV3-Small fold1 | 0.8431 | 0.7143 | 0.7735 | lightweight baseline |
| Basic CNN fold1 | 0.7327 | 0.0429 | 0.9794 | non-pretrained baseline |
| VGG16 fold1 | 0.5000 | 0.0000 | 1.0000 | baseline failure case |

The early non-timm-aware ConvNeXt-Tiny run reached AUC 0.5996 and was not used in the frozen mainline. The improved timm-aware recipe corrected the preprocessing mismatch and became the principal ConvNeXt branch.

## Why These Design Choices Were Kept

- Five-fold ensembles reduce variance under limited sample size and case-level split sensitivity.
- ConvNeXt-Tiny is the primary branch because the timm-aware recipe aligned training and inference preprocessing and produced stable ranking behavior.
- EfficientNetV2-S is retained as an auxiliary branch because its error profile is complementary and its specificity-oriented behavior stabilizes the ensemble.
- Heterogeneous preprocessing is kept because different model families respond differently to normalization, interpolation, and crop settings.
- ROI guidance is used because full-image classifiers can be influenced by borders, text overlays, probe artifacts, and background tissue.
- Largest-connected-component cropping and a non-zero margin are used to suppress fragmented masks while preserving boundary context.
- The full-image and ROI probabilities are fused in logit space because their calibration is not identical.
- The ROI area gate prevents unstable lesion crops from dominating the final operating point.
- The threshold `0.510` was selected from BUSBRA out-of-fold evidence and confirmed on locked BUSI evaluation.
- Grad-CAM and overlay outputs are aligned with ConvNeXt-Tiny so the explanation target stays fixed to the primary classifier.

## Rejected Experiments

Experiments that were tested but kept out of `configs/inference/demo.yml` include five-fold segmenters, model-zoo stacking, hard-sample weighting, seed diversity, weight soup, soft ROI gating, EfficientNet TTA, CutMix, light regularization, and 320-input retraining.

Some of these improved BUSBRA OOF metrics, but they did not transfer to BUSI or they increased deployment complexity without improving the locked external result. They remain documented in `artifacts/reports/` and are not merged into the frozen mainline.

## Repository Layout

- `configs/`: YAML files for paths, classifiers, segmenter, and inference.
- `src/datasets/`: dataset loading and split support.
- `src/models/`: classifier and segmenter factories.
- `src/engine/`: training, comparison, inference, and evaluation workflows.
- `src/explain/`: Grad-CAM and overlay generation.
- `src/preprocess/`: image I/O and preprocessing transforms.
- `src/utils/`: configuration, metrics, reports, paths, logging, and schemas.
- `scripts/`: command-line entry points.
- `app/`: Gradio web application.
- `tests/`: unit, integration, and smoke tests.
- `artifacts/reports/`: versioned experiment summaries and benchmark notes.
- `artifacts/checkpoints/`: local model weights, ignored by Git.

## Data Layout

Create `configs/paths.local.yml` from `configs/paths.example.yml` and point it to local datasets:

```yaml
datasets:
  busbra_root: ./BUSBRA
  busi_root: ./Dataset_BUSI_with_GT
```

Notes:

- Dataset folders can be renamed to match the local workstation layout.
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

- **Operating system**: Windows 10/11 x64 is the primary target.
- **Packaged desktop demo**: No local Python installation is required after unpacking the release bundle. The desktop build uses Microsoft Edge WebView2; most Windows 10/11 systems already include it.
- **CPU / GPU**: CPU inference is supported for ordinary demonstrations. NVIDIA GPU mainly improves latency.
- **Memory**: 8 GB RAM minimum, 16 GB RAM recommended.
- **Disk space**: Reserve at least 8 GB for the unpacked demo, weights, temporary files, and generated visual outputs.

### Training / Experimentation

- **Python environment**: Conda with Python 3.10 or 3.11.
- **GPU**: NVIDIA CUDA GPU is strongly recommended for full five-fold training.
- **VRAM**: 8 GB is a practical lower bound for the current 224-resolution ConvNeXt-Tiny / EfficientNetV2-S experiments; 12-16 GB is preferable.
- **System memory**: 16 GB RAM minimum, 32 GB RAM recommended.
- **Disk space**: Reserve at least 50 GB for datasets, checkpoints, logs, OOF artifacts, reports, and temporary build outputs.
- **Data handling**: Keep datasets, checkpoints, generated reports, and release bundles outside Git-tracked source files.

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

Both modes load `configs/inference/demo.yml`. The current demo uses ConvNeXt-Tiny as the primary branch and EfficientNetV2-S as the auxiliary branch.

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
- Distribute the full generated folder, not the single `.exe` file alone, because the executable depends on bundled weights, Python libraries, WebView files, and config files in the same directory.
- The `v1.1.0` release package uses the desktop-window build so the UI appears as a local Windows application instead of opening an external browser.

## Run Batch Evaluation

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_demo_convnext_effnet.json
```

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

## Key Reports

- [`artifacts/reports/native_single_model_retest.md`](artifacts/reports/native_single_model_retest.md)
- [`artifacts/reports/optimization_attempts_2026_04_27.md`](artifacts/reports/optimization_attempts_2026_04_27.md)
- [`artifacts/reports/threshold_analysis.md`](artifacts/reports/threshold_analysis.md)
- [`artifacts/reports/competition_metrics_report_regression.md`](artifacts/reports/competition_metrics_report_regression.md)
- [`artifacts/reports/competition_metrics_audit.md`](artifacts/reports/competition_metrics_audit.md)

## References

The design and optimization work drew on public breast ultrasound datasets, medical imaging open-source projects, and research on classification, segmentation, ROI-aware diagnosis, multi-task learning, and ultrasound foundation models.

- BUSI dataset: [Dataset of breast ultrasound images](https://pubmed.ncbi.nlm.nih.gov/31867417/)
- Lesion-region-aware breast ultrasound classification: [PMC11431713](https://pmc.ncbi.nlm.nih.gov/articles/PMC11431713/)
- Multi-task breast ultrasound segmentation and classification: [PMC12011763](https://pmc.ncbi.nlm.nih.gov/articles/PMC12011763/)
- OpenUS ultrasound foundation model: [XZheng0427/OpenUS](https://github.com/XZheng0427/OpenUS)
- BUSI segmentation reference project: [tqxli/breast_ultrasound_lesion_segmentation_PyTorch](https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch)
- BUSI-SAM / SAM-style segmentation references: [huangjin520/BUSI-SAM](https://github.com/huangjin520/BUSI-SAM), [bscs12/BUSSAM](https://github.com/bscs12/BUSSAM)

These references informed the data handling, model comparison, segmentation visualization, ensemble design, and ROI-aware optimization work in BUCAD.
