# LesioNeXt-BUS Design

## Project Name

**LesioNeXt-BUS**: an uncertainty-conditioned global-local network for breast ultrasound lesion diagnosis.

## Research Core

LesioNeXt couples lesion segmentation and malignancy classification through a learned
reliability weight. Its segmentation head predicts a lesion mask, a boundary map, and an
uncertainty proxy. The classification head uses global image evidence and uncertainty-
weighted lesion evidence, so unreliable ROI predictions contribute less to the final logit.

## Current Scope

- `src/models/lesionext.py` provides the unified network and a classifier-compatible wrapper.
- `configs/segmenter/lesionext.yml` and `scripts/train_lesionext.py` provide joint-training entry points.
- `configs/inference/lesionext_demo.yml` activates single-checkpoint unified inference.
- The legacy U-Net, dual-backbone ensemble, and area gate remain available as E0 baselines.

## Initial Commands

```powershell
conda run -n BUCAD python scripts/train_lesionext.py --fold 1
conda run -n BUCAD python scripts/eval_busi.py --config configs/inference/lesionext_demo.yml --output artifacts/reports/busi_lesionext_fold1.json
```

## Planned Evaluation

Compare E0 (legacy pipeline), boundary-only, reliability-only, and full LesioNeXt variants
using case-level BUSBRA OOF predictions and a frozen external evaluation protocol. Report
Dice, IoU, Boundary-F1, AUC, sensitivity, specificity, Brier score, ECE, and patient-level
bootstrap confidence intervals.
