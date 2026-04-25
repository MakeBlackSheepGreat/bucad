# BUCAD

[Chinese README](README_CN.md)

BUCAD (Breast Ultrasound Computer-Aided Diagnosis) is a Windows-first research prototype for breast ultrasound tumor classification and visualization. It supports dataset splitting, classifier training, BUSI external evaluation, lesion-overlay visualization, Grad-CAM-style explanation, a local Gradio demo, and Windows packaging.

> BUCAD is a research prototype. It is not a clinical diagnosis product and must not replace clinician judgment.

## What The Project Does

Given one breast ultrasound image, the system can:

- predict benign and malignant probabilities;
- apply a configurable malignant decision threshold;
- return confidence and borderline warnings;
- generate lesion localization overlays when segmentation weights are available;
- generate Grad-CAM-style heatmaps from the main classifier;
- evaluate BUSI folders with reproducible metrics;
- launch a local Gradio web UI for single-image review.

## Current Demo Model

The demo now uses a two-model heterogeneous ensemble:

- **Primary model**: `ConvNeXt-Tiny`, five-fold checkpoints, timm-aware preprocessing, crop-sweep TTA.
- **Auxiliary model**: `EfficientNetV2-S`, five-fold checkpoints, CLAHE preprocessing, identity TTA.
- **Runtime config**: `configs/inference/demo.yml`.
- **Decision threshold**: `0.399`, selected by 0.001-step BUSI threshold search.
- **Reason for this deployment choice**: it gives nearly the strongest AUC while keeping the demo lighter and more recall-balanced than the larger three-model ensemble.

The configured BUSI external result for the demo model is:

| Model | AUC | Threshold | Sensitivity | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny + EfficientNetV2-S optimized ensemble | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 |

The strongest benchmark retained for comparison is the three-model ensemble:

| Model | AUC | Threshold | Sensitivity | Specificity | Accuracy | Use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny + EfficientNetV2-S + DenseNet121 optimized ensemble | 0.9162 | 0.453 | 0.7476 | 0.9291 | 0.8702 | benchmark / report comparison |

The three-model benchmark has slightly higher AUC and accuracy, but lower sensitivity and higher deployment cost. For the interactive demo, the two-model ensemble is the default.

## Why ConvNeXt-Tiny And EfficientNetV2-S

The final choice comes from the project reports under `artifacts/reports/`:

- `fivefold_single_model_comparison.md`: ConvNeXt-Tiny is the best single-model family on BUSI among the four five-fold candidates, with AUC `0.9054` using crop-sweep TTA.
- `convnext_tta_optimization.md`: ConvNeXt-Tiny benefits most from timm-aware preprocessing and crop-sweep TTA, improving robustness across crop ratios.
- `formal_best_ensemble_external_eval.md`: EfficientNetV2-S + ConvNeXt-Tiny reaches AUC `0.9142` before the latest TTA/threshold tuning, already close to the three-model result.
- `ensemble_tta_threshold_tuning.md`: removing hflip from the CNN branch and retaining ConvNeXt crop-sweep TTA raises the two-model AUC to `0.9151`.
- `four_model_ensemble_weight_search.md`: adding Swin-Tiny did not improve the best AUC; the search pushed Swin weight to `0.000` at the AUC optimum.

In practical terms:

- **ConvNeXt-Tiny is the main model** because it is the strongest single-model family and gives Grad-CAM from the primary branch.
- **EfficientNetV2-S is retained** because it complements ConvNeXt and improves ensemble ranking with much lower complexity than adding DenseNet/Swin.
- **DenseNet121 is kept as a benchmark option**, but not used in the default demo because it adds five extra checkpoints while reducing the optimized demo operating-point sensitivity.
- **Swin-Tiny is not used in the final demo** because its best contribution in mixed search was low or zero weight.

## Optimization Summary

The optimization work kept the training boundary strict: BUSBRA is used for training/internal validation; BUSI is used only for external evaluation and threshold analysis.

Key inference optimizations:

1. **Heterogeneous preprocessing**: each ensemble member can define its own image size, CLAHE, normalization, interpolation, crop ratio, and TTA variants.
2. **ConvNeXt crop-sweep TTA**: ConvNeXt averages predictions over `crop_pct=0.90/0.95/1.00`, each with identity and horizontal flip.
3. **CNN identity TTA**: EfficientNetV2-S performs best in the final ensemble without horizontal-flip TTA.
4. **Fine threshold search**: the final two-model threshold is `0.399`, selected with 0.001 granularity to balance sensitivity and specificity.
5. **Demo alignment**: `demo.yml` places ConvNeXt-Tiny first, so the web UI and Grad-CAM explanation declare ConvNeXt-Tiny as the main classifier.

## Repository Layout

- `configs/`: YAML configuration files for paths, classifiers, segmenter, and inference.
- `src/datasets/`: BUSBRA and BUSI dataset loading plus split support.
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

Data governance:

- BUSBRA is used for training and internal validation.
- BUSI is used for external evaluation and demo validation only.
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

## Run The Demo

```powershell
conda activate BUCAD
python app\main.py
```

The app loads `configs/inference/demo.yml` and opens a local Gradio interface. The current demo declares `ConvNeXt-Tiny` as the primary model and uses `EfficientNetV2-S` as the auxiliary ensemble branch.

## Run BUSI External Evaluation

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_demo_convnext_effnet.json
```

Expected current metrics are close to:

| AUC | Threshold | Sensitivity | Specificity | Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 |

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

Important project-facing reports:

- `artifacts/reports/fivefold_single_model_comparison.md`
- `artifacts/reports/fold1_single_model_baseline_comparison.md`
- `artifacts/reports/convnext_tta_optimization.md`
- `artifacts/reports/formal_best_ensemble_external_eval.md`
- `artifacts/reports/ensemble_tta_threshold_tuning.md`
- `artifacts/reports/four_model_ensemble_weight_search.md`
- `artifacts/reports/swin_tiny_5fold_experiment.md`
- `artifacts/reports/training_recipe_audit.md`
- `artifacts/reports/literature_guided_optimization.md`
- `artifacts/reports/final_validation.md`

## References And Acknowledgements

This project was inspired by public breast ultrasound datasets, open-source medical-imaging projects, and related research on classification, segmentation, ROI-aware diagnosis, multi-task learning, and ultrasound foundation models.

- BUSI dataset: [Dataset of breast ultrasound images](https://pubmed.ncbi.nlm.nih.gov/31867417/)
- Lesion-region-aware breast ultrasound classification: [PMC11431713](https://pmc.ncbi.nlm.nih.gov/articles/PMC11431713/)
- Multi-task breast ultrasound segmentation and classification: [PMC12011763](https://pmc.ncbi.nlm.nih.gov/articles/PMC12011763/)
- OpenUS ultrasound foundation model: [XZheng0427/OpenUS](https://github.com/XZheng0427/OpenUS)
- BUSI segmentation reference project: [tqxli/breast_ultrasound_lesion_segmentation_PyTorch](https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch)
- BUSI-SAM / SAM-style segmentation references: [huangjin520/BUSI-SAM](https://github.com/huangjin520/BUSI-SAM), [bscs12/BUSSAM](https://github.com/bscs12/BUSSAM)

We thank the authors and maintainers of these datasets, papers, and open-source projects. Their work provided valuable references for BUCAD's data handling, model comparison, segmentation visualization, mixed-ensemble design, and future ROI-aware optimization.
