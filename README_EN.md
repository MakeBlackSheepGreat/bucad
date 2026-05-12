# BUCAD

[Chinese README](README.md)

BUCAD (Breast Ultrasound Computer-Aided Diagnosis) is a computer-aided diagnosis research prototype for benign/malignant analysis of breast ultrasound images. The system integrates deep-learning classification, semantic segmentation, ROI-guided inference, Grad-CAM explainability visualization, a local Gradio inference interface, and Windows desktop deployment.

The project is organized around two practical goals: reproducible experimentation and a runnable local demonstration. It keeps training records, external validation results, out-of-fold fusion evidence, ROI ablations, and rejected candidate protocols, while also providing a desktop demo that can be started directly after the environment and checkpoints are available. This README describes the validated stable mainline and the experimental evidence around it; exploratory branches are not treated as default deployment behavior.

The project is intended for algorithm validation, reproducible experimentation, teaching demonstrations, and controlled secondary development. It has not undergone clinical registration or multi-center clinical validation and must not be used as an independent clinical diagnosis basis. All outputs should be interpreted as algorithm research results, not clinical diagnosis conclusions.

## Data and Validation Protocol

- BUSBRA is used for model training, internal validation, out-of-fold candidate screening, and threshold selection.
- Project experiment tables report AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score.

The project follows an "internal selection, external review" boundary. The BUSBRA internal split uses case-level grouping; the recorded split contains 1875 images, 1064 unique cases, 5 folds, and leakage detection result `false`. BUSI is used as an external dataset to record transfer behavior of the mainline and candidate configurations on an independent source.

This boundary is important for breast ultrasound. Images differ in device characteristics, acquisition angle, lesion scale, background tissue, and annotation style. Internal OOF gains can overestimate transferability if they are not checked externally. The README therefore separates three categories: deployed mainline, frozen candidates reviewed externally but not merged, and internal-gain experiments that did not transfer to BUSI.

## Current Mainline Configuration

The current reproducible inference configuration is at `configs/inference/demo.yml`.

| Component | Model | Algorithm | Weight | Notes |
|---|---|---|---|---|
| Primary classifier | ConvNeXt-Tiny | ConvNeXt (Liu et al., 2022) | 0.573 | 5-fold checkpoints, timm-aware preprocessing, crop-sweep TTA |
| Auxiliary classifier | EfficientNetV2-S | EfficientNetV2 (Tan & Le, 2021) | 0.427 | 5-fold checkpoints, CLAHE preprocessing, identity TTA |
| Segmenter | UNet-ResNet18 | UNet (Ronneberger et al., 2015) + ResNet-18 encoder | - | ImageNet-pretrained encoder, binary lesion mask output |
| Fusion layer | Logistic Stacker | Logistic Regression (sklearn) | - | Trained on BUSBRA OOF predictions, using logit-space features |

The mainline inference path is: full-image classification -> segmentation-based ROI crop -> ROI classification -> logit-space fusion -> ROI quality gate -> thresholded decision -> explainability output. The full-image branch preserves global tissue context and acquisition background. The ROI branch focuses on the lesion and perilesional tissue. The stacker learns calibration relationships between these two views from BUSBRA OOF predictions. The ROI area gate falls back to full-image prediction when the predicted mask is too small or too large, reducing the effect of unreliable ROI crops.

The configuration does not use simple majority voting or manual post-hoc weighting. ConvNeXt-Tiny and EfficientNetV2-S weights come from internal OOF evidence, ROI/full fusion uses logit features rather than raw probabilities, and the classification threshold and ROI gate are determined inside BUSBRA.

### Key Technical Parameters

| Parameter | Value | Selection Basis |
|---|---|---|
| Segmentation mask threshold | 0.40 | BUSBRA Dice sweep: 0.30->0.7971, **0.40->0.8085**, 0.50->0.8074, 0.60->0.7797 |
| ROI margin ratio | 0.35 | Compromise for preserving perilesional tissue context |
| ROI area gate | [0.08, 0.75] | Falls back to full-image prediction outside range; selected by BUSBRA OOF protocol |
| Classification threshold | 0.510 | Selected from BUSBRA OOF evidence |
| Borderline marking | +/-0.08 | Samples with predicted probability within +/-0.08 of threshold are marked as uncertain |

Additional notes:

- The ConvNeXt-Tiny branch uses 5 fold checkpoints. Each member has weight 0.573 and applies CLAHE, timm mean/std, bicubic interpolation, `crop_pct=0.95`, and 6-view crop-sweep TTA.
- The EfficientNetV2-S branch uses 5 fold checkpoints. Each member has weight 0.427 and applies CLAHE, 224 input, area interpolation, and identity-only TTA. This branch provides a different convolutional inductive bias and a complementary malignant-recall tendency.
- The ROI stacker uses logit-space features with coefficients `[2.1359, 0.9337]` and intercept `-0.8671`. This indicates that full-image probability remains the main ranking source, while ROI probability supplements it as a local lesion view.
- `borderline_margin=0.08` is only a user-interface caution marker for borderline samples. It does not participate in AUC computation and does not change ranking metrics.

### BUSI External Validation Result

| Configuration | Threshold | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full mainline (ROI Area Gate) | 0.510 | 0.9256 | 0.8532 | 0.8667 | 0.8467 | 0.7309 | 0.7930 |

Confusion matrix: TN 370 / FP 67 / FN 28 / TP 182.

## System Runtime Flow

### Mainline Inference Flow

```mermaid
flowchart TD
    A["Input breast ultrasound image"] --> B["Read configs/inference/demo.yml"]
    B --> C["Load classifiers, segmenter, stacker, and threshold configuration"]
    C --> D["Image loading, channel formatting, resizing, CLAHE, and model-specific normalization"]
    D --> E["Full-image classification branch"]
    E --> E1["ConvNeXt-Tiny five folds + crop-sweep TTA"]
    E --> E2["EfficientNetV2-S five folds + identity TTA"]
    E1 --> E3["Within-family and two-model probability fusion"]
    E2 --> E3
    D --> F["Segmentation branch predicts lesion mask"]
    F --> G["Mask thresholding, largest component, ROI margin crop"]
    G --> H["ROI-image classification branch"]
    E3 --> I["Full/ROI logit features enter OOF Logistic Stacker"]
    H --> I
    F --> J["ROI area quality gate"]
    I --> J
    J --> K["Output final malignant probability"]
    K --> L["Apply decision threshold and borderline marker"]
    L --> M["Return probability, decision, mask/ROI/Grad-CAM visual outputs"]
```

1. The input image enters the shared inference service, which reads model weights, preprocessing, ROI, and threshold settings from `configs/inference/demo.yml`.
2. The full-image branch preserves global tissue context and produces a full-view malignant probability through five-fold ConvNeXt-Tiny and EfficientNetV2-S classifiers.
3. The segmentation branch predicts a lesion mask; thresholding, largest connected component extraction, and margin expansion produce an ROI crop for ROI-view classification.
4. Full-view and ROI-view probabilities are converted into logit-space features and passed to the OOF Logistic Stacker.
5. The ROI area quality gate checks whether the predicted mask is too small or too large and falls back to full-image probability when ROI evidence is abnormal.
6. The final probability is compared with the active decision threshold to produce the benign/malignant decision, borderline marker, lesion mask, ROI crop, and Grad-CAM explanation.

### Demo Workflow

```mermaid
flowchart TD
    A["Run python app\\desktop_main.py"] --> B["Start local Gradio service"]
    B --> C["Open desktop WebView or browser page"]
    C --> D["User uploads one ultrasound image"]
    D --> E["Frontend reads threshold, segmentation, and explanation options"]
    E --> F["Call BreastUltrasoundInferenceService.predict"]
    F --> G["Execute mainline inference flow"]
    G --> H["Generate malignant probability, decision, and borderline marker"]
    G --> I["Optionally generate lesion mask, ROI view, and Grad-CAM heatmap"]
    H --> J["Render structured result in the Gradio page"]
    I --> J
```

1. `app\desktop_main.py` starts the local Gradio inference service and opens the desktop or browser interface.
2. After a user uploads one image, the interface passes the image, active threshold, segmentation toggle, and explanation toggle to the inference service.
3. The service executes the full mainline flow and returns malignant probability, thresholded decision, and borderline status.
4. When visual outputs are enabled, the page also displays the lesion mask, ROI crop, and Grad-CAM heatmap.
5. The demo workflow is designed for single-image interaction and does not generate batch metrics.

### Benchmark Workflow

```mermaid
flowchart TD
    A["Run scripts/eval_busi.py --config configs/inference/demo.yml --output output_json"] --> B["Read BUSI external dataset directory"]
    B --> C["Enumerate benign / malignant images and build labels"]
    C --> D["Initialize the same inference service and mainline configuration"]
    D --> E["Run mainline inference for each image"]
    E --> F["Collect y_true and malignant_probability"]
    F --> G["Compute AUC, Accuracy, Sensitivity, Specificity, Precision, F1, and confusion matrix at the default threshold"]
    F --> H["Run threshold_sweep and Youden J analysis"]
    G --> I["Write BUSI evaluation JSON"]
    H --> I
    I --> J["Generate threshold_analysis Markdown report"]
```

1. The benchmark entry point is `scripts\eval_busi.py`; the command specifies the inference config and output JSON path.
2. The script reads the BUSI dataset directory and builds a labeled sample list from benign and malignant folders.
3. Each image uses the same inference service and `configs/inference/demo.yml` configuration as the demo path.
4. The script collects true labels and malignant probabilities for all samples, then computes fixed-threshold classification metrics and the confusion matrix.
5. It also runs threshold sweep and Youden J analysis and stores the result under `threshold_analysis`.
6. Final artifacts include the BUSI evaluation JSON, threshold-analysis Markdown report, and metric tables used by README or formal reports.

## Preprocessing Pipeline

### Image Loading and Input Normalization

Raw breast ultrasound images often contain grayscale lesion regions, device text, black borders, measurement marks, and variable frame sizes. Before entering the models, the project standardizes image loading, channel formatting, resizing, and model-specific normalization. Grayscale images are converted to three-channel inputs accepted by the classifiers; classification models use 224 input size and the segmenter uses 256 input size. Each model branch keeps its own interpolation, crop, and normalization parameters instead of forcing a single preprocessing rule that would break the distribution expected by pretrained weights.

### CLAHE Contrast Enhancement

Ultrasound images commonly exhibit low local contrast, strong speckle noise, and blurred lesion edges. The system applies Contrast Limited Adaptive Histogram Equalization (CLAHE) in configured branches to enhance lesion boundaries, internal echogenicity, and surrounding tissue contrast differences, providing more stable gray-scale structure for subsequent CNN feature extraction.

CLAHE is not a simple brightness increase. It limits the amplification strength of local histogram equalization, making low-contrast lesion borders and internal echo patterns easier for convolutional features to capture while reducing the risk of over-amplifying speckle noise. For breast ultrasound, this type of gray-scale structural enhancement is more consistent with the imaging modality than strong color perturbations or aggressive geometric transforms.

### Model-Specific Normalization

Different model families require different preprocessing due to their ImageNet pretraining configurations. The system adopts a timm-aware strategy so that mean/std, interpolation method, and crop ratio remain aligned with the pretraining recipe:

| Model | mean | std | Interpolation | crop_pct |
|---|---|---|---|---|
| ConvNeXt-Tiny | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | bicubic | 0.95 |
| EfficientNetV2-S | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | area | default |

The purpose of model-specific normalization is to keep the input distribution at inference consistent with the distribution expected during pretraining and fine-tuning. The ConvNeXt branch is aligned with the timm pretrained recipe. The EfficientNet branch uses a more conservative identity view and area interpolation to reduce inference latency and avoid unstable gains from extra views.

### Necessity of Timm-Aware Preprocessing

Timm-aware preprocessing is a hard prerequisite for effective transfer of ConvNeXt-family pretrained weights. Under the early non-timm-aware recipe tested in this project, ConvNeXt-Tiny achieved AUC of only 0.5996 on BUSI (Sensitivity=0, near-random). After introducing the timm-aware recipe, AUC improved to 0.8943 (+0.2947). Swin-Tiny similarly benefited: non-timm AUC 0.8242, timm AUC 0.8729 (+0.0487).

Detailed ablation data in `artifacts/reports/English reports/01_baseline_model_screening/native_single_model_retest.md`.

## Optimization Techniques and Ablation Studies

This section records mainline optimization in the order of first establishing model capacity and then adding stable incremental gains. All experiments follow the BUSBRA internal-selection and BUSI frozen-review boundary. A candidate is eligible for the mainline only when external AUC improves and key fixed-threshold metrics such as Sensitivity, Specificity, Precision, and F1-Score do not materially regress.

The project does not merge every seemingly advanced structure into the default demo. In small-sample medical imaging, more complex models, more checkpoints, additional TTA, or meta-learners can improve internal OOF results while degrading external transfer. The README therefore records both effective optimizations and rejection reasons so that the experimental decision path is reproducible.

The current mainline adopts the following optimization techniques: case-level five-fold cross-validation ensemble, ConvNeXt crop-sweep TTA, a UNet-ResNet18 predicted-mask ROI branch, largest connected component post-processing, ROI area quality gate, BUSBRA OOF Logistic Stacking, and the complementary two-model ensemble of ConvNeXt-Tiny and EfficientNetV2-S. These modules address model variance control, input-view augmentation, lesion-region focusing, segmentation-noise suppression, abnormal-ROI fallback, probability calibration, and architectural complementarity.

The following ablation studies summarize how each optimization affects internal validation, external review, and fixed-threshold metrics. Candidate rows in these tables are not equivalent to default deployment behavior; only modules retained after overall metric review are included in `configs/inference/demo.yml`.

### 1. Five-Fold Cross-Validation Ensemble

StratifiedGroupKFold with case-level grouping is used to ensure that samples from the same patient do not appear across folds. During inference, outputs are first aggregated within each model family across five folds and then passed to the downstream fusion stage.

The five-fold strategy addresses two sources of instability. First, single-fold training is sensitive to case assignment when medical-image sample size is limited. Second, multiple fold checkpoints form a lightweight ensemble at inference time and reduce dependence on one local training split. ConvNeXt-Tiny, EfficientNetV2-S, DenseNet-121, and Swin-Tiny all showed higher external AUC after five-fold aggregation, indicating that the strategy is useful across different model families.

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

EfficientNetV2-S gains the most from five-fold aggregation (+0.0388), suggesting stronger single-fold variance. ConvNeXt-Tiny is already relatively stable as a single fold, but still gains +0.0111 AUC. This supports using within-family five-fold averaging as the base layer instead of selecting only one apparently strong fold.

Detailed data in `artifacts/reports/English reports/01_baseline_model_screening/fivefold_single_model_comparison.md`.

### 2. Crop-Sweep Test-Time Augmentation

Lesion size, position, and surrounding tissue background vary significantly across breast ultrasound images. A single center crop may either lose perilesional tissue by cropping too tightly or introduce excessive irrelevant background by cropping too loosely.

The ConvNeXt branch uses three crop ratios (0.90, 0.95, 1.00), each with horizontal flipping, producing six inference views. Predictions are averaged across all views to reduce variance caused by any single crop scale.

This TTA design is applied to the ConvNeXt branch rather than all model families. ConvNeXt showed more stable benefit from crop-view variation. Extending multi-view inference to EfficientNetV2-S increases latency, while the internal gain was not strong enough to justify the cost. The deployment configuration therefore allocates TTA budget to the branch where it provides the clearest external benefit.

**Ablation results (ConvNeXt-Tiny 5-fold, BUSI external):**

| TTA Strategy | AUC | Sensitivity | Specificity | F1-Score |
|---|---:|---:|---:|---:|
| identity (baseline) | 0.8991 | 0.7381 | 0.8810 | 0.7434 |
| + horizontal flip | 0.9041 | - | 0.8924 | 0.7524 |
| + crop-sweep | **0.9054** | **0.7714** | 0.8719 | **0.7696** |
| + rotation +/-5 deg | 0.9048 | - | **0.8970** | - |

Crop-sweep achieves the best AUC (+0.0063), Sensitivity (+0.0333), and F1 (+0.0262) over identity baseline.

Rotation +/-5 degrees improves Specificity at some threshold points, but does not exceed the overall benefit of crop-sweep. Because probe angle and lesion orientation can carry clinical information, stronger rotation also risks changing local texture and shape expression. The default mainline therefore uses crop-sweep and excludes rotation TTA.

Detailed data in `artifacts/reports/English reports/04_tta_threshold_external_eval/convnext_tta_optimization.md`.

### 3. ROI Segmentation Guidance

Full-image classifiers receive the entire ultrasound frame, which may include borders, device annotations, probe regions, and normal tissue textures. The ROI branch uses a semantic segmentation model (UNet + ResNet-18 encoder) to predict a lesion mask, then crops the ROI region after largest connected component extraction and margin expansion, and evaluates lesion-focused malignant probability with the classifier.

The ROI branch is not intended to displace the full-image branch. It provides a complementary local lesion view. The full image preserves acquisition context and surrounding tissue, while the ROI image suppresses non-lesion regions. After stacker fusion, the two views can be balanced per sample. To avoid the non-deployable upper bound caused by ground-truth masks, formal candidate selection uses predicted masks only; Oracle ROI is retained only as a theoretical control.

Segmenter-related optimization is focused on deployable mask post-processing and ROI geometry constraints, not on using manual ground-truth masks as inference input. The project scans mask thresholds on BUSBRA segmentation validation and selects `0.40`, the threshold with the highest Dice. It then extracts the largest connected component from predicted masks to suppress fragmented regions and false positives near annotations or device marks. ROI cropping applies `margin_ratio=0.35` to retain perilesional tissue and acoustic-shadow context. All ROI/full fusion experiments use OOF caches generated from predicted masks, avoiding GT-mask leakage during candidate selection.

**Segmentation and ROI post-processing evidence:**

| Segmentation / ROI Processing | Internal Basis | Downstream Effect |
|---|---|---|
| Mask threshold 0.30 | BUSBRA Dice 0.7971, ROI tends to be large | More background can enter ROI crops |
| Mask threshold 0.40 | BUSBRA Dice **0.8085**, best in this sweep | Default mask binarization threshold |
| Mask threshold 0.50 | BUSBRA Dice 0.8074, close to 0.40 | Slightly lower Dice and less weak-boundary retention |
| Mask threshold 0.60 | BUSBRA Dice 0.7797 | Threshold too high, may remove low-echo borders |
| Largest connected component | Reduces fragmented-mask interference | BUSI AUC 0.9196 -> 0.9208, Sensitivity 0.8429 -> 0.8524 |
| ROI margin 0.35 | Preserves perilesional context | Avoids overly tight crops that lose morphology and surrounding tissue cues |

**Ablation results (BUSI external):**

| Configuration | AUC | Sensitivity | Note |
|---|---:|---:|---|
| Full-image baseline | 0.9151 | - | Dual-model ensemble, no ROI |
| + Segmenter ROI | 0.9196 | 0.8429 | +0.0045 |
| + LCC post-processing | 0.9208 | 0.8524 | +0.0057 vs baseline |
| Oracle ROI (GT mask) | 0.9202 | - | Theoretical upper-bound control, not deployable |

ROI guidance provides a stable +0.0057 AUC improvement. LCC post-processing further improves AUC by +0.0012 and Sensitivity by +0.0095 by suppressing fragmented masks.

This result shows that ROI benefit comes from deployable predicted masks rather than information leakage from manual masks. Oracle ROI is not substantially higher than predicted ROI, which also suggests that the remaining classification bottleneck is not determined by segmentation overlap alone. ROI crop scale, classifier viewpoint, and probability calibration all contribute to downstream behavior.

Detailed data in `artifacts/reports/English reports/02_roi_segmentation/roi_oof_experiment.md` and `artifacts/reports/English reports/02_roi_segmentation/roi_oof_lcc_optimization.md`.

### 4. ROI Area Quality Gate

Segmentation predictions are not always reliable. Area too small (< 0.08) may indicate the segmenter captured only noise; area too large (> 0.75) means the mask nearly covers the entire image, losing the focusing benefit. The area gate falls back to full-image prediction when ROI quality is abnormal.

The area gate is motivated by error-case analysis. Some benign cases lose surrounding tissue context after ROI cropping and are pushed toward malignant probability by local texture. Some predicted masks cover only a tiny region or almost the whole frame, indicating that the ROI is no longer trustworthy. The gate uses an interpretable and reproducible rule to identify these cases and lets the full-image branch take over when ROI evidence is weak.

**Ablation results (BUSI external):**

| Configuration | AUC | Sensitivity | Specificity | Precision | F1-Score | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| ROI OOF LCC (baseline) | 0.9208 | 0.8524 | 0.8169 | 0.6911 | 0.7633 | 80 | 31 |
| + Area gate | **0.9256** | **0.8667** | **0.8467** | **0.7309** | **0.7930** | 67 | 28 |
| **Improvement** | **+0.0048** | **+0.0143** | **+0.0297** | **+0.0398** | **+0.0297** | **-13** | **-3** |

In this ablation group, the area gate is the only technique that simultaneously improves all six metrics. Rejected alternatives include threshold-only tuning (insufficient gain), removing LCC (AUC/Sensitivity regression), and gate < 0.25 (AUC -0.0116).

The key benefit of the area gate is that it reduces both FP and FN: FP decreases from 80 to 67, and FN decreases from 31 to 28. For benign/malignant auxiliary diagnosis, this is more valuable than improving only one fixed-threshold metric, because the gate is not simply trading sensitivity for specificity through a threshold shift. It improves branch selection on samples with unreliable ROI evidence.

Detailed data in `artifacts/reports/English reports/02_roi_segmentation/roi_precision_f1_study.md`.

### 5. OOF Logistic Stacking

Probability distributions differ across models, TTA views, and ROI/full-image branches; direct averaging can amplify systematic bias from one branch. The system trains a Logistic Regression fusion model on BUSBRA out-of-fold predictions, using logit-space features as input, avoiding direct weight search on the external validation set.

OOF training is used to prevent a sample from being used both to fit a base model and to train the fusion model on that model's in-sample prediction. Each OOF probability is produced by a fold model that did not see that sample during training, which better matches deployment behavior. Logit-space fusion is more suitable than direct probability-space linear mixing because probabilities are compressed near 0 and 1, while logits retain calibration information for the linear stacker.

**Ablation results (BUSI external):**

| Fusion Method | BUSI AUC | Note |
|---|---:|---|
| Static weighted average | 0.9130 | eff 0.427 / conv 0.573 |
| OOF Logistic Stacking | 0.9138 | +0.0008, multi-view input |

OOF stacking provides a more disciplined internal protocol for training a fusion model, but external AUC improvement is marginal (+0.0008). Its primary value is learning branch weights and calibration relationships from internal OOF predictions rather than manually tuning them on the external validation set.

In this project, stacking is therefore positioned as a disciplined fusion and calibration mechanism rather than a standalone large-AUC module. It becomes more useful when combined with ROI/full-view calibration and the ROI area gate.

Detailed data in `artifacts/reports/English reports/03_ensemble_oof_stacking/oof_two_model_stacking.md`.

### 6. Ensemble Member Selection

The project systematically screened multiple candidate models. The following are representative single-fold (fold1) external review results; pretrained classifiers such as ConvNeXt, Swin, DenseNet, and EfficientNet use the timm-aware recipe, while YOLO26x-cls is an independent side experiment:

Model selection is not decided by AUC alone. The project considers external AUC, Sensitivity/Specificity balance, Precision/F1, internal validation stability, five-fold gain, inference cost, checkpoint count, complementarity of error patterns, and overfitting risk on a small dataset. The final mainline selects ConvNeXt-Tiny and EfficientNetV2-S because the two branches have complementary inductive biases: ConvNeXt-Tiny provides stronger overall ranking ability, while EfficientNetV2-S contributes a different CNN-family view and malignant-recall tendency.

| Model | Params | AUC | Sensitivity | Specificity | F1-Score |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 28M | 0.8943 | 0.7762 | 0.8741 | 0.7617 |
| ConvNeXt-Small | 50M | 0.8947 | 0.7667 | 0.8581 | 0.7436 |
| Swin-Tiny | 28M | 0.8729 | 0.7048 | 0.8902 | 0.7291 |
| DenseNet-121 | 8M | 0.8766 | 0.4571 | 0.9771 | 0.6076 |
| EfficientNetV2-S | 21M | 0.8609 | 0.8333 | 0.7048 | 0.6809 |
| ResNet-18 | 11M | 0.8480 | 0.7714 | 0.8124 | 0.7137 |
| MobileNetV3-Small | 2.5M | 0.8431 | 0.7143 | 0.7735 | 0.6536 |
| YOLO26x-cls | 28M | 0.8189 | 0.6571 | 0.8650 | 0.6781 |
| Basic CNN | - | 0.7327 | 0.0429 | 0.9794 | 0.0789 |
| VGG-16 | 138M | 0.5000 | 0.0000 | 1.0000 | 0.0000 |

The table shows that parameter count does not directly determine performance. VGG-16 has the largest parameter count but fails to transfer. Basic CNN has high Specificity but almost cannot detect malignant samples. DenseNet-121 provides high Precision/Specificity but insufficient Sensitivity. EfficientNetV2-S has high single-fold Sensitivity but lower Specificity. As a side classification check, YOLO26x-cls reached BUSBRA fold1 val AUC 0.8216 and BUSI external AUC 0.8189, below the existing ConvNeXt/Swin/DenseNet single-fold baselines. ConvNeXt-Tiny is preferred because its metrics are more balanced and because its earlier failure was resolved after correcting the timm-aware preprocessing mismatch.

**Two-model vs three-model ensemble (BUSI formal external evaluation):**

| Configuration | AUC | Sensitivity | Specificity | Accuracy | F1-Score |
|---|---:|---:|---:|---:|---:|
| Two-model (ConvNeXt + EfficientNet) | 0.9142 | 0.7524 | 0.9130 | 0.8609 | 0.7783 |
| Three-model (+ DenseNet) | 0.9144 | 0.7095 | 0.9382 | 0.8640 | 0.7720 |
| **Difference** | **+0.0002** | **-0.0429** | **+0.0252** | **+0.031** | **-0.0063** |

Three-model AUC exceeds two-model by only 0.0002, but Sensitivity drops 4.3% with increased deployment complexity (15 vs 10 checkpoints). Two-model achieves higher Youden-optimal Accuracy (0.8655 vs 0.8516), so the mainline selects two models.

This reflects the project merge rule: a tiny AUC gain is not sufficient if malignant recall decreases and deployment complexity grows. For a competition demo and a medical-image auxiliary diagnosis workflow, model choice needs to balance performance, stability, and explainable engineering complexity.

Detailed data in `artifacts/reports/English reports/03_ensemble_oof_stacking/formal_best_ensemble_external_eval.md`.

### 7. ConvNeXt-Small Upgrade Evaluation

ConvNeXt-Small (50M params) shows marginally higher single-fold external AUC than ConvNeXt-Tiny (28M params), but lower internal AUC and training loss near zero, indicating overfitting risk under the current training recipe.

ConvNeXt-Small is a reasonable upgrade candidate, but it is not yet a default-configuration merge item. Its single-fold external AUC is higher by 0.0038, but internal fold1 AUC is lower by 0.0141 and Youden J is nearly identical. This combination of a small external single-point gain and weaker internal stability does not meet the mainline merge criterion. Further work on this direction should first complete stricter five-fold training, regularization, and OOF transfer validation.

**Ablation results:**

| Model | BUSBRA fold1 AUC | BUSI fold1 AUC | Youden J |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.9259 | 0.8953 | 0.6671 |
| ConvNeXt-Small | 0.9118 | 0.8991 | 0.6665 |
| **Difference** | **-0.0141** | **+0.0038** | **-0.0006** |

The contradiction between internal AUC decrease (-0.0141) and external AUC increase (+0.0038) indicates unstable generalization under the current training configuration. Youden J is essentially identical (0.6671 vs 0.6665), not meeting the mainline merge criterion.

Detailed data in `artifacts/reports/English reports/01_baseline_model_screening/convnext_small_upgrade_experiment.md`.

### 8. Threshold Selection and Metric Trade-Offs

The project reports both AUC and fixed-threshold metrics. AUC reflects ranking ability and is independent of a specific threshold. Sensitivity, Specificity, Precision, and F1-Score describe the practical behavior at a decision point. The mainline threshold `0.510` comes from BUSBRA OOF evidence and the fixed review protocol.

Threshold tuning was tested as an independent direction, but moving only the threshold changes the FP/FN distribution without improving probability ranking quality. The ROI area gate improves both AUC and fixed-threshold metrics, indicating that it changes branch selection and probability quality rather than only applying a post-hoc threshold shift.

### 9. Explainability Output

The system retains Grad-CAM and lesion-mask visualization to show the relationship between classifier attention and segmentation ROI. Explainability output is not used for training or parameter tuning. It mainly supports demo presentation, error-case review, and manual inspection. For misclassified samples, Grad-CAM helps determine whether the model attends to the lesion itself, surrounding tissue, black borders, or device annotations, supporting later error-type attribution.

## Tested but Rejected Approaches

| Approach | BUSBRA OOF | BUSI External | Rejection Reason |
|---|---|---|---|
| Model-Zoo OOF candidates | AUC 0.9352 | AUC 0.9213 | Large internal OOF gain, but external AUC and Sensitivity decreased; likely OOF overfitting |
| Hard-sample focal retraining | fold1 AUC 0.9051 | Not sent to BUSI | Over-focused on difficult samples; ranking quality below original ConvNeXt fold1 |
| Mild hard-sample retraining | fold1 AUC 0.9086 | Not sent to BUSI | Sample weighting reduced AUC and did not meet five-fold training standard |
| Area-aware dynamic weights | AUC 0.9232 | AUC 0.9185 | Did not exceed mainline 0.9208 |
| OOF Meta-Learner | AUC 0.9241 | AUC 0.9115 | External AUC regression of 0.0093 |
| ROI soft gate + multi-scale | AUC 0.9221 | AUC 0.9189 | External AUC below mainline 0.9256 |
| ROI area gate OOF protocol | AUC 0.9233 | - | Candidate did not exceed mainline |
| Non-0.40 mask thresholds | Dice below 0.8085 | Not merged | 0.40 has the best BUSBRA segmentation Dice; lower thresholds enlarge ROI, higher thresholds lose weak borders |
| Removing LCC | - | AUC 0.9196 | Below LCC post-processing AUC 0.9208, with Sensitivity regression |
| ConvNeXt seed diversity | AUC 0.9249 | AUC 0.9250 | External AUC slightly below mainline and Sensitivity decreased |
| Weight Soup (seed 42+123) | AUC 0.9232 | AUC 0.9225 | Weight averaging reduced runtime cost, but AUC and F1 decreased |
| Light regularization retraining | fold1 AUC 0.9096 | Not sent to BUSI | AUC too far below the original fold1 |
| CutMix / Mixup | fold1 AUC 0.9145 | Not sent to BUSI | Specificity improved, but ranking quality and Sensitivity decreased |
| 320 input resolution | fold1 AUC 0.9210 | Not sent to BUSI | Increased VRAM cost, Sensitivity decreased, and original fold1 was not exceeded |
| EfficientNet TTA | AUC 0.9190 | Not sent to BUSI | Internal gain was too small and inference latency increased |
| YOLO26x-cls single fold | AUC 0.8216 | AUC 0.8189 | Side classification candidate; did not exceed existing ConvNeXt/Swin/DenseNet single-fold baselines |

Detailed records are retained in the corresponding categorized protocol files under `artifacts/reports/English reports/`.

These rejected experiments share a common pattern: internal OOF or single-fold metrics can improve locally, but the improvement does not transfer consistently to external validation. The project therefore uses strict merge conditions to avoid putting complex but non-transferable methods into the default demo. This also explains why the mainline remains conservative: in BUSI external review, a simple, stable, and interpretable ROI area gate was more reliable than more complicated post-hoc fusion schemes.

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

This cumulative path reflects the actual sources of mainline improvement. The largest gains come from correcting timm-aware preprocessing, five-fold aggregation, two-model complementarity, and the ROI area gate. OOF stacking has a small standalone AUC gain, but it provides a disciplined fusion protocol. ROI segmentation guidance depends on post-processing and quality gating, so downstream classification performance cannot be judged from Dice alone.

From an engineering perspective, the final mainline does not assume that more models are always better. Each additional module is required to produce an interpretable error reduction. The area gate reduces both FP and FN, crop-sweep improves ConvNeXt external AUC and F1, and five-fold aggregation reduces fold variance; these benefits are directly traceable in the ablation tables.

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

`artifacts/reports/Chinese reports/` and `artifacts/reports/English reports/` store reports by experiment topic, making it possible to trace evidence from model screening, ROI segmentation, OOF fusion, TTA/threshold analysis, error analysis, and demo release. This README lists only the key mainline-related reports; finer-grained failed experiments and side candidates remain in the corresponding subdirectories.

`.pt` weights under `artifacts/checkpoints/` are managed through Git LFS. If a teammate clones the repository and checkpoint files are only a few KB, the files are LFS pointers rather than real weights and `git lfs pull` is required. The default demo depends on the ConvNeXt-Tiny five-fold checkpoints, EfficientNetV2-S five-fold checkpoints, and the segmenter checkpoint.

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

The main runtime environment is the `BUCAD` conda environment. Windows local demo usage only requires inference dependencies; training and batch evaluation additionally require CUDA, PyTorch, timm, segmentation_models_pytorch, scikit-learn, OpenCV, pandas, PyYAML, tqdm, Gradio, and Grad-CAM dependencies. For demo-only use, first verify:

1. `conda activate BUCAD` enters the environment normally.
2. `python check_env.py` detects PyTorch, OpenCV, timm, and other major dependencies.
3. `.pt` files under `artifacts/checkpoints/` are real checkpoint files rather than Git LFS pointers.

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

The desktop demo reads `configs/inference/demo.yml` by default. This frozen mainline uses ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate for single-image inference and explainability output. The demo outputs malignant probability, thresholded decision, borderline-sample marker, lesion mask, ROI crop, and Grad-CAM heatmap. If segmentation or explainability image generation fails, the service can fall back within the configured behavior and still return the base classification probability.

For browser-only use, run:

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

Before development merges, run unit tests:

```powershell
conda activate BUCAD
python -m pytest tests/unit -q
```

Unit tests cover metric computation, ROI inference, descriptor features, segmentation modules, and basic numerical stability. They cannot substitute for BUSI external validation, but they can catch interface, shape, and basic logic errors before submission.

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

`scripts/eval_busi.py` is the unified BUSI external-review entry point. Its JSON output contains fixed-threshold metrics and threshold analysis. After a candidate configuration is prepared internally, this script can generate its external review result.

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
| `01_baseline_model_screening/yolo_cls_yolo26x-cls_fold1.md` | YOLO26x-cls fold1 side classification comparison |
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
