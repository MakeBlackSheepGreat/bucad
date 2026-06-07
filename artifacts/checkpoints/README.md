# Runtime Checkpoints

The main desktop demo uses `configs/inference/demo.yml` and expects the following Git LFS checkpoint files:

| Component | Files |
|---|---|
| ConvNeXt-Tiny classifier | `convnext_tiny_timm_recipe_fold1.pt` to `convnext_tiny_timm_recipe_fold5.pt` |
| EfficientNetV2-S classifier | `efficientnetv2_s_fold1.pt` to `efficientnetv2_s_fold5.pt` |
| U-Net segmenter | `segmenter_fold1.pt` |

After cloning, use:

```powershell
git lfs pull
conda activate BUCAD
python app\desktop_main.py
```

Only the runtime checkpoints above are tracked by Git LFS. Historical experiment
weights may remain in this local directory for reproducibility, but `.gitignore`
keeps them out of the default repository handoff unless they are promoted into
the runtime configuration.
