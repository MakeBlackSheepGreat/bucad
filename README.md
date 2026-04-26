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

The current mainline is `ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF LCC`, configured in `configs/inference/demo.yml`.

- **Primary branch**: `ConvNeXt-Tiny`, five-fold checkpoints, ensemble weight `0.573`, timm-aware preprocessing, and crop-sweep TTA. It is also the declared main model for the web UI and Grad-CAM explanations.
- **Auxiliary branch**: `EfficientNetV2-S`, five-fold checkpoints, ensemble weight `0.427`, CLAHE preprocessing, and identity TTA. It complements ConvNeXt-Tiny while keeping deployment lighter than adding DenseNet/Swin.
- **ROI branch**: `segmenter_fold1.pt` predicts a lesion mask; BUSBRA validation selects `mask_threshold=0.40`, `margin_ratio=0.35`, and largest-connected-component ROI cropping.
- **OOF fusion**: the system predicts both full-image probability and ROI probability, then combines them with a BUSBRA OOF-trained logit logistic stacker.
- **Decision threshold**: `0.550`, selected from BUSBRA ROI out-of-fold predictions, not from BUSI.
- **Data boundary**: BUSBRA is used for training, internal validation, OOF model selection, ROI parameter selection, and threshold selection. BUSI is used only for locked external evaluation after the configuration is fixed.

The configured BUSI external result for the demo model is:

| Model | AUC | Threshold | Sensitivity | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF Stacking + LCC ROI | 0.9208 | 0.550 | 0.8524 | 0.8169 | 0.8284 |

The strongest non-ROI benchmark retained for comparison is the three-model ensemble:

| Model | AUC | Threshold | Sensitivity | Specificity | Accuracy | Use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny + EfficientNetV2-S + DenseNet121 optimized ensemble | 0.9162 | 0.453 | 0.7476 | 0.9291 | 0.8702 | benchmark / report comparison |

The three-model benchmark is kept as a non-ROI comparison, but the ROI-guided two-model stack now has the stronger locked external AUC and lower deployment complexity than a 15-checkpoint three-model ensemble.

## Why ConvNeXt-Tiny And EfficientNetV2-S

The current demo candidate is documented by the project reports under `artifacts/reports/`. BUSI numbers in those reports are external observations after a configuration is fixed; they must not be used as the sole basis for training, tuning, model-family selection, ensemble-weight selection, or threshold selection.

- `fivefold_single_model_comparison.md`: records the external BUSI behavior of the four five-fold candidates; ConvNeXt-Tiny is the strongest observed single-model candidate with crop-sweep TTA.
- `convnext_tta_optimization.md`: ConvNeXt-Tiny benefits most from timm-aware preprocessing and crop-sweep TTA, improving robustness across crop ratios.
- `formal_best_ensemble_external_eval.md`: EfficientNetV2-S + ConvNeXt-Tiny reaches AUC `0.9142` before the latest TTA/threshold tuning, already close to the three-model result.
- `ensemble_tta_threshold_tuning.md`: removing hflip from the CNN branch and retaining ConvNeXt crop-sweep TTA raises the two-model AUC to `0.9151`.
- `roi_oof_experiment.md` and `roi_oof_lcc_optimization.md`: BUSBRA OOF-trained ROI stacking plus BUSBRA-selected LCC mask post-processing raises the locked BUSI external AUC to `0.9208`.
- `four_model_ensemble_weight_search.md`: adding Swin-Tiny did not improve the best AUC; the search pushed Swin weight to `0.000` at the AUC optimum.

In practical terms, the current demo keeps the lightweight two-model candidate, but the competition-final model should be locked from BUSBRA OOF/internal validation before any final external evaluation:

- **ConvNeXt-Tiny is the main model** because it is the strongest single-model family and gives Grad-CAM from the primary branch.
- **EfficientNetV2-S is retained** because it complements ConvNeXt and improves ensemble ranking with much lower complexity than adding DenseNet/Swin.
- **DenseNet121 is kept as a benchmark option**, but not used in the default demo because it adds five extra checkpoints while reducing the optimized demo operating-point sensitivity.
- **Swin-Tiny is not used in the final demo** because its best contribution in mixed search was low or zero weight.

## Optimization Summary

The optimization work keeps the data boundary strict: BUSBRA is used for training, internal validation, OOF model selection, and threshold selection. BUSI is used only for locked external evaluation after the configuration is fixed.

Key inference optimizations:

1. **Heterogeneous preprocessing**: each ensemble member can define its own image size, CLAHE, normalization, interpolation, crop ratio, and TTA variants.
2. **ConvNeXt crop-sweep TTA**: ConvNeXt uses ImageNet mean/std, bicubic interpolation, and `crop_pct=0.90/0.95/1.00`; each crop evaluates identity and horizontal flip.
3. **EfficientNet identity branch**: EfficientNetV2-S uses CLAHE, area interpolation, no ConvNeXt mean/std inheritance, and identity TTA to avoid cross-model preprocessing mismatch.
4. **ROI OOF stacking**: the final demo combines full-image probability and ROI probability with a BUSBRA OOF-trained logit logistic stacker.
5. **ROI mask post-processing**: BUSBRA segmentation validation selects mask threshold `0.40`, largest connected component cropping, and `margin_ratio=0.35` before ROI inference.
6. **OOF threshold selection**: the final ROI stack threshold is `0.550`, selected on BUSBRA out-of-fold predictions to avoid tuning on BUSI.
7. **Demo alignment**: `demo.yml` places ConvNeXt-Tiny first, so the web UI and Grad-CAM explanation declare ConvNeXt-Tiny as the main classifier.

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
| 0.9208 | 0.550 | 0.8524 | 0.8169 | 0.8284 |

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
