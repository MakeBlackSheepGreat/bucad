# Training Recipe Audit

Date: 2026-04-25

## Question

Do the previously trained models share the same training-recipe problems found
in the quick Swin-Tiny and ConvNeXt-Tiny experiments?

## Short Answer

Partly yes.

The previous timm-based models were trained with the project’s generic
classifier recipe, not with each model family’s full recommended timm fine-tune
recipe. This affects how strongly we can judge the absolute potential of each
backbone. However, it does not invalidate the project’s data boundary,
case-level split, or the fact that the existing EfficientNetV2-S and DenseNet121
models achieved useful results under this project recipe.

## Shared Recipe Limitations

| Area | Current Project Behavior | Why It Matters |
| --- | --- | --- |
| Normalization | Images are scaled to `[0, 1]` only | Pretrained timm backbones usually expect ImageNet or model-specific mean/std |
| Model-specific transform | One shared resize pipeline is used | timm models have different interpolation/crop/input-size defaults |
| Scheduler/warmup | Fixed learning rate AdamW | Transformer/ConvNeXt-style models often need warmup and cosine decay |
| Checkpoint selection | Default is last checkpoint | Best AUC / best Youden checkpoint can be better for small medical datasets |
| Hyperparameter reuse | Same simple recipe reused across families | Good for quick screening, weak for judging a model family’s ceiling |

## Evidence From Current Code

- The classifier transform resizes, optionally applies CLAHE, scales by 255, and
  converts to CHW tensor.
- The training loop uses AdamW with a fixed learning rate and default
  `checkpoint_strategy = last`.
- EfficientNetV2-S, DenseNet121, Swin-Tiny, and ConvNeXt-Tiny were all tested
  with the project-level training pipeline rather than a model-specific timm
  recipe.

## Model-Specific Difference

The issue is not equally severe for every model.

| Model Family | Current Evidence | Risk Interpretation |
| --- | --- | --- |
| EfficientNetV2-S | Strong fold and BUSI results already exist | Recipe is not fully optimal, but the model clearly works in this project |
| DenseNet121 | Strong fold results and useful ensemble complement | Also not fully optimized, but useful under current recipe |
| ResNet18 / MobileNetV3 | Reasonable fold-1 baselines | Useful as fair project-recipe baselines |
| Swin-Tiny | High sensitivity but low specificity | Current recipe likely unsuitable; cannot judge Swin family ceiling |
| ConvNeXt-Tiny | Collapsed to all-benign at threshold 0.50 | Strong sign that current recipe is bad for this run |
| Basic CNN / VGG16 | Weak under current recipe | Not serious final candidates without major recipe changes |

## What Is Still Valid

- BUSBRA/BUSI data boundary remains valid.
- BUSI was not used for training.
- Case-level fold split remains valid.
- The fold-1 basic comparison is still useful as a "same project recipe"
  comparison.
- The final mixed ensemble result is still a valid empirical result under the
  implemented inference pipeline.

## What Should Be Downgraded

- Do not claim Swin-Tiny or ConvNeXt-Tiny are inherently unsuitable.
- Do not claim the current fold-1 basic comparison is each model’s best possible
  performance.
- Do not compare quick model-family screening results directly against the final
  optimized ensemble.
- Do not treat 224-only and fixed-LR results as final evidence for models whose
  recommended recipes differ substantially.

## Recommended Fix

Add a second-level "timm recommended recipe" experiment path:

1. Use `timm.data.resolve_model_data_config()` or equivalent config extraction
   for model-specific input size, interpolation, crop policy, mean, and std.
2. Add configurable normalization: `none`, `imagenet`, `timm`, or `custom`.
3. Add scheduler support: warmup + cosine decay.
4. Add explicit best checkpoint selection: best AUC, best Youden, or
   sensitivity-constrained score.
5. Run fold1 only first for EfficientNetV2-S, DenseNet121, Swin-Tiny, and
   ConvNeXt-Tiny under the improved recipe.
6. Only run five folds if fold1 is competitive.

## Practical Conclusion

The earlier results are not useless, but they should be labeled correctly:

- "Project recipe v1" results: fair for comparing models under the current
  implemented pipeline.
- "Model ceiling" results: not yet available for Swin/ConvNeXt and not fully
  available for other timm backbones.

The next serious step is to implement a timm-aware training recipe and rerun
fold1 for the strongest existing and new candidate backbones.
