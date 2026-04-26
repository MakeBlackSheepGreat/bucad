# Full Model Comparison Summary

Date: 2026-04-24

Command: `conda run -n BUCAD python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20`

Dry run: `False`
Model count: `7`

## Results

| Rank | Model | Status | AUC | Sensitivity | Specificity | Accuracy | Runtime Seconds | Notes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `tf_efficientnetv2_s` | completed | 0.8937 | 0.7049 | 0.8775 | 0.8213 | 342.1 | EfficientNetV2-S main model |
| 2 | `densenet121` | completed | 0.8867 | 0.7213 | 0.8775 | 0.8267 | 638.4 | DenseNet-121 baseline |
| 3 | `resnet18` | completed | 0.8756 | 0.7131 | 0.8577 | 0.8107 | 153.9 | ResNet-18 baseline |
| 4 | `mobilenetv3_small_100` | completed | 0.8704 | 0.6557 | 0.9170 | 0.8320 | 188.2 | MobileNetV3-Small baseline |
| 5 | `basic_cnn` | completed | 0.6427 | 0.0000 | 0.9921 | 0.6693 | 90.6 | Basic CNN baseline without pretrained weights |
| 6 | `vgg16` | completed | 0.5000 | 0.0000 | 1.0000 | 0.6747 | 707.2 | VGG-16 baseline |
| - | `alexnet` | failed | - | - | - | - | 0.3 | Unknown model (alexnet) in recorded run; torchvision fallback support was added afterward |

## Conclusion

- `tf_efficientnetv2_s` achieved the highest fold-1 validation AUC among completed comparison models.
- `densenet121`, `resnet18`, and `mobilenetv3_small_100` are strong secondary baselines.
- `alexnet` failed during this recorded run before torchvision fallback support was added; rerun the comparison if an updated AlexNet metric is required.
- This comparison supports freezing EfficientNetV2-S as the final classifier family.

## External Evaluation Context

- BUSI ensemble AUC: `0.8955`
- Selected threshold: `0.25`
- Selected-threshold sensitivity/specificity/accuracy: `0.8476` / `0.8215` / `0.8300`
