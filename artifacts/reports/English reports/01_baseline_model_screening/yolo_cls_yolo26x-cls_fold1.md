# YOLO Fold1 Single-Model Comparison

## Experiment Boundary

- This is a side model-screening experiment and does not modify `configs/inference/demo.yml`.
- Training uses BUSBRA fold1 train only; BUSBRA fold1 val is used for internal review.
- BUSI is used once for frozen external review and is not used for threshold feedback or further tuning.
- Only one YOLO classification candidate is tested in this run.

## Model Selection

- YOLO candidate: `yolo26x-cls.pt`.
- Selection basis: in the public Ultralytics YOLO26 classification family, the x-scale classification checkpoint is the strongest public classification weight by reported Top-1 accuracy.
- Task adaptation: the model is fine-tuned as a benign/malignant image classifier. Detection boxes and segmentation masks are not used.

## Training Setup

- fold: `1`
- epochs: `30`
- imgsz: `224`
- batch: `4`
- device: `0`
- training samples: `1500`
- BUSBRA val samples: `375`
- BUSI samples: `647`
- best checkpoint: `artifacts/yolo_runs/yolo26x_cls_fold1/weights/best.pt`

## Results

| Dataset | Threshold | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BUSBRA fold1 val | 0.50 | 0.8216 | 0.7973 | 0.6230 | 0.8814 | 0.7170 | 0.6667 | TN 223 / FP 30 / FN 46 / TP 76 |
| BUSI external | 0.50 | 0.8189 | 0.7975 | 0.6571 | 0.8650 | 0.7005 | 0.6781 | TN 378 / FP 59 / FN 72 / TP 138 |

## Youden-Optimal Point

These thresholds are reported for analysis only and are not fed back into model selection or external tuning.

| Dataset | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Youden J | Confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BUSBRA fold1 val | 0.50 | 0.7973 | 0.6230 | 0.8814 | 0.7170 | 0.6667 | 0.5044 | TN 223 / FP 30 / FN 46 / TP 76 |
| BUSI external | 0.49 | 0.7975 | 0.6619 | 0.8627 | 0.6985 | 0.6797 | 0.5246 | TN 377 / FP 60 / FN 71 / TP 139 |

## Fold1 Reference Baselines

BUSI external review at the default threshold 0.50:

| Model | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Small timm recipe fold1 | 0.8947 | 0.8284 | 0.7667 | 0.8581 | 0.7220 | 0.7436 |
| ConvNeXt-Tiny timm recipe fold1 | 0.8943 | 0.8423 | 0.7762 | 0.8741 | 0.7477 | 0.7617 |
| DenseNet121 fold1 | 0.8766 | 0.8083 | 0.4571 | 0.9771 | 0.9057 | 0.6076 |
| Swin-Tiny timm recipe fold1 | 0.8729 | 0.8300 | 0.7048 | 0.8902 | 0.7551 | 0.7291 |
| EfficientNetV2-S fold1 | 0.8609 | 0.7465 | 0.8333 | 0.7048 | 0.5757 | 0.6809 |
| yolo26x-cls fold1 | 0.8189 | 0.7975 | 0.6571 | 0.8650 | 0.7005 | 0.6781 |

## Conclusion

- YOLO26x-cls fold1 does not exceed the existing ConvNeXt single-fold baselines and is not recommended for the mainline.
- This result only reflects the base potential of a single-fold classification model. It does not represent a five-fold ensemble or ROI-aware mainline configuration.
