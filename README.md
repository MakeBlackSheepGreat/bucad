# BUCAD

[Chinese README](README_CN.md)

BUCAD (Breast Ultrasound Computer-Aided Diagnosis) is a computer-aided diagnosis research prototype for benign/malignant analysis of breast ultrasound images. The system integrates deep-learning classification, semantic segmentation, ROI-guided inference, Grad-CAM explainability visualization, a local Gradio inference interface, and Windows desktop deployment.

The project is intended for algorithm validation, reproducible experimentation, teaching demonstrations, and controlled secondary development. It has not undergone clinical registration or multi-center clinical validation and must not be used as an independent clinical diagnosis basis.

## Data and Validation Protocol

- BUSBRA is used for model training, internal validation, out-of-fold candidate screening, and threshold selection.
- BUSI is used only after candidates are frozen, for locked external validation and result review. It does not participate in training, model selection, parameter search, or threshold tuning.
- Project experiment tables report AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score.

## Current Mainline Configuration

The current reproducible inference configuration is at `configs/inference/demo.yml`.

| Component | Model | Algorithm | Weight | Notes |
|---|---|---|---|---|
| Primary classifier | ConvNeXt-Tiny | ConvNeXt (Liu et al., 2022) | 0.573 | 5-fold checkpoints, timm-aware preprocessing, crop-sweep TTA |
| Auxiliary classifier | EfficientNetV2-S | EfficientNetV2 (Tan & Le, 2021) | 0.427 | 5-fold checkpoints, CLAHE preprocessing, identity TTA |
| Segmenter | UNet-ResNet18 | UNet (Ronneberger et al., 2015) + ResNet-18 encoder | - | ImageNet-pretrained encoder, binary lesion mask output |
| Fusion layer | Logistic Stacker | Logistic Regression (sklearn) | - | Trained on BUSBRA OOF predictions, using logit-space features |

### Key Technical Parameters

| Parameter | Value | Selection Basis |
|---|---|---|
| Segmentation mask threshold | 0.40 | BUSBRA Dice sweep: 0.30->0.7971, **0.40->0.8085**, 0.50->0.8074, 0.60->0.7797 |
| ROI margin ratio | 0.35 | Compromise for preserving perilesional tissue context |
| ROI area gate | [0.08, 0.75] | Falls back to full-image prediction outside range; selected by BUSBRA OOF protocol |
| Classification threshold | 0.510 | Selected from BUSBRA OOF evidence; BUSI is used only for frozen external review |
| Borderline marking | +/-0.08 | Samples with predicted probability within +/-0.08 of threshold are marked as uncertain |

### BUSI External Validation Result

| Configuration | Threshold | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full mainline (ROI Area Gate) | 0.510 | 0.9256 | 0.8532 | 0.8667 | 0.8467 | 0.7309 | 0.7930 |

Confusion matrix: TN 370 / FP 67 / FN 28 / TP 182.

## Preprocessing Pipeline

### CLAHE Contrast Enhancement

Ultrasound images commonly exhibit low local contrast, strong speckle noise, and blurred lesion edges. The system applies Contrast Limited Adaptive Histogram Equalization (CLAHE) in configured branches to enhance lesion boundaries, internal echogenicity, and surrounding tissue contrast differences, providing more stable gray-scale structure for subsequent CNN feature extraction.

### Model-Specific Normalization

Different model families require different preprocessing due to their ImageNet pretraining configurations. The system adopts a timm-aware strategy so that mean/std, interpolation method, and crop ratio remain aligned with the pretraining recipe:

| Model | mean | std | Interpolation | crop_pct |
|---|---|---|---|---|
| ConvNeXt-Tiny | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | bicubic | 0.95 |
| EfficientNetV2-S | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | area | default |

### Necessity of Timm-Aware Preprocessing

Timm-aware preprocessing is a hard prerequisite for effective transfer of ConvNeXt-family pretrained weights. Under the early non-timm-aware recipe, ConvNeXt-Tiny achieved AUC of only 0.5996 on BUSI (Sensitivity=0, near-random). After introducing the timm-aware recipe, AUC improved to 0.8943 (+0.2947). Swin-Tiny similarly benefited: non-timm AUC 0.8242, timm AUC 0.8729 (+0.0487).

Detailed ablation data in `artifacts/reports/English reports/01_baseline_model_screening/native_single_model_retest.md`.

## Optimization Techniques and Ablation Studies

### 1. Five-Fold Cross-Validation Ensemble

StratifiedGroupKFold with case-level grouping is used to ensure that samples from the same patient do not appear across folds. During inference, outputs are first aggregated within each model family across five folds and then passed to the downstream fusion stage.

**Ablation results (BUSBRA 5-fold CV AUC):**

| Model | Fold1 Single | 5-fold Mean | Std | Note |
|---|---:|---:|---:|---|
| ConvNeXt-Tiny V1 | 0.9259 | 0.9212 | 0.0196 | - |
| ConvNeXt-Tiny V2 | 0.9278 | 0.9176 | 0.0107 | More stable |
| ConvNeXt-Small | 0.9118 | 0.9162 | 0.0163 | - |
| EfficientNetV2-S | 0.9248 | 0.8946 | 0.0241 | - |

**BUSI external ablation (5-fold vs single fold):**

| Model | Single Fold AUC | 5-fold AUC | Improvement |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.8943 | 0.9054 | +0.0111 |
| EfficientNetV2-S | 0.8609 | 0.8997 | +0.0388 |
| DenseNet-121 | 0.8766 | 0.8914 | +0.0148 |
| Swin-Tiny | 0.8729 | 0.8971 | +0.0242 |

Detailed data in `artifacts/reports/English reports/01_baseline_model_screening/fivefold_single_model_comparison.md`.

### 2. Crop-Sweep Test-Time Augmentation

Lesion size, position, and surrounding tissue background vary significantly across breast ultrasound images. A single center crop may either lose perilesional tissue by cropping too tightly or introduce excessive irrelevant background by cropping too loosely.

The ConvNeXt branch uses three crop ratios (0.90, 0.95, 1.00), each with horizontal flipping, producing six inference views. Predictions are averaged across all views to reduce variance caused by any single crop scale.

**Ablation results (ConvNeXt-Tiny 5-fold, BUSI external):**

| TTA Strategy | AUC | Sensitivity | Specificity | F1-Score |
|---|---:|---:|---:|---:|
| identity (baseline) | 0.8991 | 0.7381 | 0.8810 | 0.7434 |
| + horizontal flip | 0.9041 | - | 0.8924 | 0.7524 |
| + crop-sweep | **0.9054** | **0.7714** | 0.8719 | **0.7696** |
| + rotation +/-5 deg | 0.9048 | - | **0.8970** | - |

Crop-sweep achieves the best AUC (+0.0063), Sensitivity (+0.0333), and F1 (+0.0262) over identity baseline.

Detailed data in `artifacts/reports/English reports/04_tta_threshold_external_eval/convnext_tta_optimization.md`.

### 3. ROI Segmentation Guidance

Full-image classifiers receive the entire ultrasound frame, which may include borders, device annotations, probe regions, and normal tissue textures. The ROI branch uses a semantic segmentation model (UNet + ResNet-18 encoder) to predict a lesion mask, then crops the ROI region after largest connected component extraction and margin expansion, and evaluates lesion-focused malignant probability with the classifier.

**Ablation results (BUSI external):**

| Configuration | AUC | Sensitivity | Note |
|---|---:|---:|---|
| Full-image baseline | 0.9151 | - | Dual-model ensemble, no ROI |
| + Segmenter ROI | 0.9196 | 0.8429 | +0.0045 |
| + LCC post-processing | 0.9208 | 0.8524 | +0.0057 vs baseline |
| Oracle ROI (GT mask) | 0.9202 | - | Theoretical upper-bound control, not deployable |

ROI guidance provides a stable +0.0057 AUC improvement. LCC post-processing further improves AUC by +0.0012 and Sensitivity by +0.0095 by suppressing fragmented masks.

Detailed data in `artifacts/reports/English reports/02_roi_segmentation/roi_oof_experiment.md` and `artifacts/reports/English reports/02_roi_segmentation/roi_oof_lcc_optimization.md`.

### 4. ROI Area Quality Gate

Segmentation predictions are not always reliable. Area too small (< 0.08) may indicate the segmenter captured only noise; area too large (> 0.75) means the mask nearly covers the entire image, losing the focusing benefit. The area gate falls back to full-image prediction when ROI quality is abnormal.

**Ablation results (BUSI external):**

| Configuration | AUC | Sensitivity | Specificity | Precision | F1-Score | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| ROI OOF LCC (baseline) | 0.9208 | 0.8524 | 0.8169 | 0.6911 | 0.7633 | 80 | 31 |
| + Area gate | **0.9256** | **0.8667** | **0.8467** | **0.7309** | **0.7930** | 67 | 28 |
| **Improvement** | **+0.0048** | **+0.0143** | **+0.0297** | **+0.0398** | **+0.0297** | **-13** | **-3** |

In this ablation group, the area gate is the only technique that simultaneously improves all six metrics. Rejected alternatives include threshold-only tuning (insufficient gain), removing LCC (AUC/Sensitivity regression), and gate < 0.25 (AUC -0.0116).

Detailed data in `artifacts/reports/English reports/02_roi_segmentation/roi_precision_f1_study.md`.

### 5. OOF Logistic Stacking

Probability distributions differ across models, TTA views, and ROI/full-image branches; direct averaging can amplify systematic bias from one branch. The system trains a Logistic Regression fusion model on BUSBRA out-of-fold predictions, using logit-space features as input, avoiding direct weight search on the external validation set.

**Ablation results (BUSI external):**

| Fusion Method | BUSI AUC | Note |
|---|---:|---|
| Static weighted average | 0.9130 | eff 0.427 / conv 0.573 |
| OOF Logistic Stacking | 0.9138 | +0.0008, multi-view input |

OOF stacking provides a more disciplined internal protocol for training a fusion model, but external AUC improvement is marginal (+0.0008). Its primary value is learning branch weights and calibration relationships from internal OOF predictions rather than manually tuning them on the external validation set.

Detailed data in `artifacts/reports/English reports/03_ensemble_oof_stacking/oof_two_model_stacking.md`.

### 6. Ensemble Member Selection

The project systematically screened multiple candidate models. The following are representative single-fold (fold1) timm-aware recipe external review results:

| Model | Params | AUC | Sensitivity | Specificity | F1-Score |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 28M | 0.8943 | 0.7762 | 0.8741 | 0.7617 |
| ConvNeXt-Small | 50M | 0.8947 | 0.7667 | 0.8581 | 0.7436 |
| Swin-Tiny | 28M | 0.8729 | 0.7048 | 0.8902 | 0.7291 |
| DenseNet-121 | 8M | 0.8766 | 0.4571 | 0.9771 | 0.6076 |
| EfficientNetV2-S | 21M | 0.8609 | 0.8333 | 0.7048 | 0.6809 |
| ResNet-18 | 11M | 0.8480 | 0.7714 | 0.8124 | 0.7137 |
| MobileNetV3-Small | 2.5M | 0.8431 | 0.7143 | 0.7735 | 0.6536 |
| Basic CNN | - | 0.7327 | 0.0429 | 0.9794 | 0.0789 |
| VGG-16 | 138M | 0.5000 | 0.0000 | 1.0000 | 0.0000 |

**Two-model vs three-model ensemble (BUSI formal external evaluation):**

| Configuration | AUC | Sensitivity | Specificity | Accuracy | F1-Score |
|---|---:|---:|---:|---:|---:|
| Two-model (ConvNeXt + EfficientNet) | 0.9142 | 0.7524 | 0.9130 | 0.8609 | 0.7783 |
| Three-model (+ DenseNet) | 0.9144 | 0.7095 | 0.9382 | 0.8640 | 0.7720 |
| **Difference** | **+0.0002** | **-0.0429** | **+0.0252** | **+0.031** | **-0.0063** |

Three-model AUC exceeds two-model by only 0.0002, but Sensitivity drops 4.3% with increased deployment complexity (15 vs 10 checkpoints). Two-model achieves higher Youden-optimal Accuracy (0.8655 vs 0.8516), so the mainline selects two models.

Detailed data in `artifacts/reports/English reports/03_ensemble_oof_stacking/formal_best_ensemble_external_eval.md`.

### 7. ConvNeXt-Small Upgrade Evaluation

ConvNeXt-Small (50M params) shows marginally higher single-fold external AUC than ConvNeXt-Tiny (28M params), but lower internal AUC and training loss near zero, indicating overfitting risk under the current training recipe.

**Ablation results:**

| Model | BUSBRA fold1 AUC | BUSI fold1 AUC | Youden J |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.9259 | 0.8953 | 0.6671 |
| ConvNeXt-Small | 0.9118 | 0.8991 | 0.6665 |
| **Difference** | **-0.0141** | **+0.0038** | **-0.0006** |

The contradiction between internal AUC decrease (-0.0141) and external AUC increase (+0.0038) indicates unstable generalization under the current training configuration. Youden J is essentially identical (0.6671 vs 0.6665), not meeting the replacement criterion.

Detailed data in `artifacts/reports/English reports/01_baseline_model_screening/convnext_small_upgrade_experiment.md`.

## Tested but Rejected Approaches

| Approach | BUSBRA OOF | BUSI External | Rejection Reason |
|---|---|---|---|
| Area-aware dynamic weights | AUC 0.9232 | AUC 0.9185 | Did not exceed mainline 0.9208 |
| OOF Meta-Learner | AUC 0.9241 | AUC 0.9115 | External AUC regression of 0.0093 |
| ROI soft gate + multi-scale | AUC 0.9221 | AUC 0.9189 | External AUC below mainline 0.9256 |
| ROI area gate OOF protocol | AUC 0.9233 | - | Candidate did not exceed mainline |
| Weight Soup (seed 42+123) | - | - | Internal gains did not transfer externally |
| CutMix / Mixup | - | - | No stable improvement on small dataset |
| 320 input resolution | - | - | Increased VRAM, no external AUC improvement |
| EfficientNet TTA | - | - | Marginal internal gain, increased inference latency |

Detailed records are retained in the corresponding categorized protocol files under `artifacts/reports/English reports/`.

## Cumulative Pipeline Improvement

The following shows the BUSI external AUC accumulation path from baseline to final configuration:

| Stage | Configuration | BUSI AUC | Cumulative Gain |
|---|---|---:|---:|
| Single-fold ConvNeXt-Tiny | fold1, identity | 0.8943 | Baseline |
| + timm-aware recipe | Corrected preprocessing mismatch | 0.8943 | Prerequisite |
| + 5-fold ensemble | 5-fold average | 0.9054 | +0.0111 |
| + crop-sweep TTA | 3 crop x hflip | 0.9054 | Included in 5-fold |
| + EfficientNetV2-S auxiliary | Dual-model static weighting | 0.9130 | +0.0076 |
| + OOF Logistic Stacking | Logit fusion | 0.9138 | +0.0008 |
| + ROI segmentation guidance | UNet + LCC | 0.9208 | +0.0070 |
| + Area quality gate | [0.08, 0.75] fallback | **0.9256** | **+0.0048** |
| **Total improvement** | | | **+0.0313** |

## Metric Definitions

| Metric | Definition | Clinical Meaning |
|---|---|---|
| AUC | Area under ROC curve | Overall ranking ability for benign/malignant discrimination |
| Accuracy | (TP+TN) / (TP+TN+FP+FN) | Overall prediction correctness |
| Sensitivity (Recall) | TP / (TP+FN) | Malignant detection rate, reflects missed diagnosis risk |
| Specificity | TN / (TN+FP) | Benign correct identification rate, reflects false positive control |
| Precision | TP / (TP+FP) | Positive prediction reliability |
| F1-Score | 2PR / (P+R) | Harmonic mean of Precision and Recall |
| Youden J | Sensitivity + Specificity - 1 | Composite threshold quality assessment |
| Dice | 2 * overlap / (area(A) + area(B)) | Segmentation mask overlap with ground truth |

## Repository Layout

```
BUCAD/
├── configs/                          # Configuration files
│   ├── classifier/                   # Classifier training configs
│   ├── segmenter/                    # Segmenter training configs
│   └── inference/                    # Inference and ensemble configs
├── src/
│   ├── datasets/                     # Dataset loading and splits
│   ├── models/                       # Model factories (classifier/segmenter)
│   ├── engine/                       # Training/evaluation/inference engine
│   ├── explain/                      # Grad-CAM explainability
│   ├── preprocess/                   # Image preprocessing and ROI cropping
│   └── utils/                        # Config/metrics/reports/logging
├── scripts/                          # Command-line entry points
├── app/                              # Gradio web application
├── packaging/                        # Windows desktop packaging
├── tests/                            # Unit/integration/smoke tests
└── artifacts/
    ├── checkpoints/                  # Model weights (managed through Git LFS or local assets)
    └── reports/                      # Experiment reports and evaluation results
        ├── Chinese reports/          # Chinese experiment reports, categorized by experiment topic
        ├── English reports/          # English experiment reports, categorized by experiment topic
        └── README.md                 # Report directory taxonomy
```

## Running

### Environment Setup

After cloning the repository, install dependencies and ensure Git LFS checkpoints are hydrated:

```powershell
git lfs install
git lfs pull

conda create -n BUCAD python=3.11 -y
conda activate BUCAD
python -m pip install -r requirements.txt
```

The main demo weights referenced by `configs/inference/demo.yml` are stored under `artifacts/checkpoints/` through Git LFS. They include the 5-fold ConvNeXt-Tiny checkpoints, the 5-fold EfficientNetV2-S checkpoints, and `segmenter_fold1.pt`. A normal clone with Git LFS installed should download them automatically; if the `.pt` files are only a few KB, they are pointer files and `git lfs pull` is required.

Training or batch evaluation requires local dataset paths. Copy `configs/paths.example.yml` to `configs/paths.local.yml` and point to the actual dataset locations:

```yaml
datasets:
  busbra_root: ./训练集/BUSBRA
  busi_root: ./测试集/Dataset_BUSI_with_GT
```

### Run Demo Application

```powershell
conda activate BUCAD
python app\desktop_main.py
```

The desktop demo reads `configs/inference/demo.yml` by default. This frozen mainline uses ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate for single-image inference and explainability output. For browser-only use, run:

```powershell
conda activate BUCAD
python app\main.py
```

### Run Test Programs

```powershell
conda activate BUCAD
python check_env.py
python check_all.py
```

`check_env.py` checks Python, CUDA, PyTorch, and major dependencies. `check_all.py` runs project-level environment and basic functionality checks. After these checks pass, `python app\desktop_main.py` should start the local demo interface.

### Windows Packaging

```powershell
python -m PyInstaller --clean --noconfirm packaging\desktop_demo.spec
```

Output: `dist/bucad-demo-desktop/bucad-demo-desktop.exe`. Distribute the full directory.

### Training and Evaluation

```powershell
# Generate data splits
python scripts\make_split.py --config configs\paths.local.yml

# Train classifier (single fold)
python scripts\train_cls.py --config configs\classifier\convnext_tiny_timm_recipe.yml --fold 1

# Batch evaluation
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_demo.json
```

## Hardware Requirements

| Item | Inference / Demo | Training / Experiment |
|---|---|---|
| OS | Windows 10/11 x64 | Windows 10/11 x64 |
| Python | Not needed after packaging | Conda + 3.11 |
| GPU | Not required | NVIDIA CUDA, 8GB+ VRAM |
| Memory | 8GB min, 16GB recommended | 16GB min, 32GB recommended |
| Disk | 8GB | 50GB |

## Key Experiment Report Index

| Report | Content |
|---|---|
| `01_baseline_model_screening/native_single_model_retest.md` | Timm-aware vs non-timm-aware recipe comparison |
| `01_baseline_model_screening/fivefold_single_model_comparison.md` | Four-model 5-fold vs single-fold comparison |
| `04_tta_threshold_external_eval/convnext_tta_optimization.md` | ConvNeXt TTA strategy ablation |
| `02_roi_segmentation/roi_oof_experiment.md` | ROI guidance vs full-image comparison |
| `02_roi_segmentation/roi_oof_lcc_optimization.md` | LCC post-processing ablation |
| `02_roi_segmentation/roi_precision_f1_study.md` | ROI area gate ablation |
| `03_ensemble_oof_stacking/oof_two_model_stacking.md` | OOF Stacking vs static weights |
| `03_ensemble_oof_stacking/formal_best_ensemble_external_eval.md` | Two-model vs three-model formal evaluation |
| `01_baseline_model_screening/convnext_small_upgrade_experiment.md` | ConvNeXt-Small vs Tiny comparison |
| `01_baseline_model_screening/six_model_comparison_report.md` | Six-model comprehensive comparison (BUSBRA + BUSI) |

Full report directory: `artifacts/reports/English reports/`; taxonomy is documented in `artifacts/reports/README.md`.

## References

- Al-Dhabyani W, et al. Dataset of breast ultrasound images. *Data in Brief*, 2020. [[PubMed]](https://pubmed.ncbi.nlm.nih.gov/31867417/)
- Liu Z, et al. A ConvNet for the 2020s. *CVPR*, 2022.
- Tan M, Le Q. EfficientNetV2: Smaller models and faster training. *ICML*, 2021.
- Ronneberger O, et al. U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*, 2015.
- He K, et al. Deep residual learning for image recognition. *CVPR*, 2016.
- ROI-aware breast ultrasound classification. [[PMC11431713]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11431713/)
- Multi-task breast ultrasound segmentation and classification. [[PMC12011763]](https://pmc.ncbi.nlm.nih.gov/articles/PMC12011763/)
- OpenUS ultrasound foundation model. [[GitHub]](https://github.com/XZheng0427/OpenUS)
- BUSI segmentation reference. [[GitHub]](https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch)
- BUSI-SAM / BUSSAM segmentation references. [[GitHub]](https://github.com/huangjin520/BUSI-SAM) [[GitHub]](https://github.com/bscs12/BUSSAM)

## License

This project is for research, teaching, and competition reproducibility only.
