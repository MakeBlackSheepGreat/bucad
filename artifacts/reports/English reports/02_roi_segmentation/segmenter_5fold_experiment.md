# Segmenter 5-Fold Experiment

Date: 2026-04-26 22:59:21

## Setup

- Config: `configs/segmenter/unet_5fold.yml`
- Architecture: U-Net with `resnet18` encoder
- Image size: `256`
- Epochs per fold: `5`
- Output checkpoints: `artifacts/checkpoints/segmenter_5fold_fold{fold}.pt`

## Dice Summary

- Mean Dice: `0.8675`
- Std Dice: `0.0170`
- Best fold: `5` (`0.8852`)
- Weakest fold: `2` (`0.8383`)

| Fold | Dice | Train Size | Val Size | Checkpoint |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.8593 | 1500 | 375 | `artifacts\checkpoints\segmenter_5fold_fold1.pt` |
| 2 | 0.8383 | 1500 | 375 | `artifacts\checkpoints\segmenter_5fold_fold2.pt` |
| 3 | 0.8747 | 1500 | 375 | `artifacts\checkpoints\segmenter_5fold_fold3.pt` |
| 4 | 0.8800 | 1500 | 375 | `artifacts\checkpoints\segmenter_5fold_fold4.pt` |
| 5 | 0.8852 | 1500 | 375 | `artifacts\checkpoints\segmenter_5fold_fold5.pt` |

## Interpretation

- This run does not overwrite the frozen demo segmenter `artifacts/checkpoints/segmenter_fold1.pt`.
- The 5-fold checkpoints are ready for ROI-mask stability analysis and possible segmentation ensemble experiments.
- Before replacing the runtime segmenter, evaluate whether the 5-fold masks improve ROI Area Gate metrics on BUSBRA OOF and then run one locked BUSI external check.
