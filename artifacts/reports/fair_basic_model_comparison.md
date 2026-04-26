# Basic Model Fairness Review

> ???????????? BUSI ??????????????????? BUSI ???????TTA?????????????????????????????????? BUSBRA OOF/????????? BUSI ????????


Date: 2026-04-25

## Purpose

This report re-organizes the existing model test results under a fairer
comparison rule. The goal is to understand the basic performance of each model
family, not to compare a raw fold-1 model against the final optimized ensemble.

## Fair Comparison Rule

A result is treated as a strict basic-model comparison only when it satisfies
all of the following conditions:

- Same dataset role: BUSBRA for training and validation only.
- Same split: fold 1, 1500 training images and 375 validation images.
- Same input recipe: 224 image size, CLAHE, horizontal flip augmentation.
- Same training budget: 20 epochs.
- Same decision threshold: 0.50 for Sensitivity, Specificity, and Accuracy.
- Same model form: one single model, no five-fold ensemble, no mixed ensemble,
  no threshold tuning, no TTA-only enhancement comparison.

Under this rule, the most reliable basic comparison source is the fold-1 model
comparison run.

## Strict Fold-1 Basic Comparison

| Rank | Model | AUC | Sensitivity | Specificity | Accuracy | Basic Finding |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | EfficientNetV2-S | 0.8937 | 0.7049 | 0.8775 | 0.8213 | Best AUC among strict single-model baselines |
| 2 | DenseNet121 | 0.8867 | 0.7213 | 0.8775 | 0.8267 | Very close to EfficientNet, slightly higher sensitivity and accuracy |
| 3 | ResNet18 | 0.8756 | 0.7131 | 0.8577 | 0.8107 | Stable and lightweight strong baseline |
| 4 | MobileNetV3-Small | 0.8704 | 0.6557 | 0.9170 | 0.8320 | Best accuracy/specificity among compact baselines, lower sensitivity |
| 5 | Basic CNN | 0.6427 | 0.0000 | 0.9921 | 0.6693 | Too weak; almost no malignant recall at threshold 0.50 |
| 6 | VGG16 | 0.5000 | 0.0000 | 1.0000 | 0.6747 | Failed to learn useful discrimination in this run |
| - | AlexNet | - | - | - | - | Recorded run failed before fallback support was added |

## Interpretation

The strict comparison shows that the useful baseline group is actually clear:
EfficientNetV2-S, DenseNet121, ResNet18, and MobileNetV3-Small are all in the
0.87 to 0.89 AUC range. EfficientNetV2-S is the best ranking model by AUC, but
DenseNet121 is close enough to be a reasonable complementary model. This
supports the later decision to use EfficientNetV2-S as the main backbone and
DenseNet121 as an ensemble partner.

MobileNetV3-Small is not the best by AUC, but it has high specificity and the
highest accuracy in the strict table. If the project later needs a lightweight
CPU-friendly demo model, MobileNetV3-Small is a better candidate than Basic CNN
or VGG16.

Basic CNN and VGG16 should not be treated as serious final candidates under the
current recipe. Their apparent accuracy is misleading because the validation
set has more benign than malignant samples; both models miss essentially all
malignant samples at the default threshold.

## Extended Fold-1 Reference Results

The following models were tested after the original comparison run. They are
useful for screening, but they are not part of the strict original comparison
table because they were launched separately and, in the Swin case, used a
different learning rate.

| Model | AUC | Sensitivity | Specificity | Accuracy | Difference From Strict Rule | Finding |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Swin-Tiny | 0.7950 | 0.8852 | 0.5020 | 0.6267 | lr=0.0001, separate run | High sensitivity but many false positives; not worth five-fold under this recipe |
| ConvNeXt-Tiny | 0.4519 | 0.0000 | 1.0000 | 0.6747 | separate run | Collapsed to all-benign prediction at threshold 0.50 |

These results do not mean Transformer or ConvNeXt families can never work. They
only show that the current quick recipes are not competitive with the existing
EfficientNetV2-S / DenseNet121 / ResNet18 / MobileNetV3 group.

Important limitation: Swin-Tiny and ConvNeXt-Tiny were not trained with their
full model-family recommended recipes. They reused the project’s simple
classifier training pipeline for a quick screening run. In particular, the
current pipeline does not apply each timm model's pretrained normalization
statistics, crop policy, scheduler, warmup, layer-wise learning-rate decay, or
model-specific augmentation recipe. Therefore these two results should be read
as "quick project-recipe screening results", not as the upper bound of Swin or
ConvNeXt.

## Results That Should Not Be Mixed Into Basic Backbone Ranking

The following results are useful, but they answer different questions and
should not be mixed into the basic single-model ranking:

| Result Type | Example | Why It Is Not A Basic Backbone Comparison |
| --- | --- | --- |
| Longer EfficientNet training | EfficientNetV2-S fold1 AUC 0.9248 | 30-epoch final training, not the same 20-epoch comparison budget |
| Five-fold model evidence | EfficientNetV2-S five-fold mean AUC 0.8946 | Multi-fold robustness result, not one fold architecture screening |
| Resolution experiments | 256 / 320 EfficientNetV2-S runs | Tests input size and augmentation, not only backbone choice |
| Sensitivity tuning | Sensitive / balanced-sensitive EfficientNet runs | Changes class weighting and checkpoint selection objective |
| BUSI TTA / threshold results | TTA and threshold sweep reports | Tests inference operating point, not training backbone alone |
| Mixed ensembles | EfficientNetV2-S + DenseNet121 final ensemble | Combines multiple models and thresholds; stronger but not a basic model |

## Practical Conclusion

For basic model performance, the current evidence supports this ordering:

1. EfficientNetV2-S is the best main backbone by fold-1 AUC.
2. DenseNet121 is the closest strong partner and has complementary behavior.
3. ResNet18 and MobileNetV3-Small are strong secondary baselines.
4. Swin-Tiny is currently high-recall but low-specificity, so it is not a good
   final candidate without a better training recipe.
5. ConvNeXt-Tiny, Basic CNN, and VGG16 are not usable under the tested recipes.

The fairest next model experiment would not be another final-ensemble
comparison. It should be a new strict fold-1 single-model comparison where any
new candidate uses the same split, image size, epoch budget, augmentation, and
threshold reporting as the original comparison table. If the goal is to judge
Swin or ConvNeXt more seriously, run a second-level comparison with timm
pretrained normalization, best-checkpoint selection, lower learning rate,
warmup/scheduler, and class-balance control.
