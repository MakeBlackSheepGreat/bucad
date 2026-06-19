# SonoGloReNet-T fold1 Benchmark Report

> Data boundary: BUSBRA is used for training, internal validation, and model selection only. BUSI is used as the locked external evaluation set. This report uses a single-fold `fold1` comparison against the current `ConvNeXt-Tiny timm recipe fold1` baseline.

Date: 2026-06-18

## Objective

- Verify that `SonoGloReNet-T` trains stably inside the current classification pipeline.
- Compare its internal validation and BUSI external generalization against `ConvNeXt-Tiny timm recipe fold1`.
- Check whether the new structure introduces meaningful inference overhead.

## Model Setup

### SonoGloReNet-T

- Training config: `configs/classifier/sonoglore_convnext_tiny.yml`
- Inference config: `configs/inference/sonoglore_convnext_tiny_fold1.yml`
- Structure:
  - `ConvNeXt-Tiny features_only`
  - one lightweight self-attention block on `stage4`
  - three-scale `1x1 Conv + GELU + GeM` pooling on `stage2/3/4`
  - `concat + LayerNorm + MLP` classification head

### Baseline

- Training report: `artifacts/reports/train_cls_convnext_tiny_timm_recipe_fold1.json`
- External evaluation report: `artifacts/reports/busi_convnext_tiny_timm_recipe_fold1.json`

## Training Results

| Model | fold | Internal AUC | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny timm recipe | 1 | 0.9259 | 0.7213 | 0.9328 | 0.8640 | - | - | 13 |
| SonoGloReNet-T | 1 | 0.9096 | 0.7295 | 0.9012 | 0.8453 | 0.7807 | 0.7542 | 22 |

## BUSI External Evaluation

Default threshold `0.50`:

| Model | AUC | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny timm recipe fold1 | 0.8926 | 0.7762 | 0.8696 | 0.8393 | 0.7409 | 0.7581 | TN 380 / FP 57 / FN 47 / TP 163 |
| SonoGloReNet-T fold1 | 0.8782 | 0.8048 | 0.7963 | 0.7991 | 0.6550 | 0.7222 | TN 348 / FP 89 / FN 41 / TP 169 |

Best threshold by Youden J:

| Model | AUC | Suggested threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Youden J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny timm recipe fold1 | 0.8926 | 0.54 | 0.7667 | 0.8879 | 0.8485 | 0.7667 | 0.7667 | 0.6545 |
| SonoGloReNet-T fold1 | 0.8782 | 0.72 | 0.7810 | 0.8421 | 0.8223 | 0.7039 | 0.7404 | 0.6231 |

## Inference Latency

Test setup:

- Environment: `BUCAD`
- Device: runtime auto-selected device
- Input: the same BUSI benign/malignant sample image
- Settings: segmentation and Grad-CAM disabled, classification only
- Statistics: 1 warm-up run, then 5 measured runs

| Model | Mean latency (s) | Min latency (s) | Max latency (s) |
| --- | ---: | ---: | ---: |
| ConvNeXt-Tiny timm recipe fold1 | 0.01785 | 0.01591 | 0.02216 |
| SonoGloReNet-T fold1 | 0.01771 | 0.01712 | 0.01810 |

Observation: in this run, `SonoGloReNet-T` does not introduce visible extra inference cost for single-image classification.

## Analysis

- `SonoGloReNet-T` trains successfully and completes BUSI external evaluation, which confirms that the architecture integration is functionally correct.
- However, the current `fold1` result does not improve external AUC over the baseline.
- Relative to `ConvNeXt-Tiny timm recipe fold1`:
  - internal validation AUC decreases by about `0.0164`
  - BUSI AUC decreases by about `0.0144`
  - external sensitivity increases slightly from `0.7762` to `0.8048`
  - external specificity drops from `0.8696` to `0.7963`
- This suggests that the current version is more recall-oriented for malignant cases, but it also produces more false positives and weaker overall ranking/calibration than the baseline.

## Current Conclusion

- Based on `fold1`, `SonoGloReNet-T v1` does not outperform the current `ConvNeXt-Tiny timm recipe` baseline.
- The direction is still worth keeping because inference cost remains essentially unchanged and recall improves slightly.
- The practical conclusion for now is that this is a research branch worth further tuning, not a ready replacement for the current mainline backbone.

## Recommended Next Steps

1. Complete `SonoGloReNet-T 5-fold` training to confirm whether the current result is just fold-level variance.
2. Try a stronger regularized version first:
   - `drop_path_rate`
   - `label_smoothing`
   - `MixUp / CutMix`
   - a more conservative learning rate
3. Run ablations on the `stage4 attention` block:
   - remove attention and keep only multi-scale pooling
   - keep only `stage3 + stage4`
4. If the `5-fold` result still trails the baseline, keep this model as a research branch and do not promote it into the current mainline ensemble.
