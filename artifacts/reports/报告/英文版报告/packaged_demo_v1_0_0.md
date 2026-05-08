<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->
# BUCAD v1.0.0 Windows Demo Packaging Record

Date: 2026-04-26

## Version Positioning

`v1.0.0` is a Windows demo version for competition presentation, aimed at allowing judges or teammates to launch a local Gradio diagnostic interface via a single executable.

## Packaging Command

```powershell
conda activate BUCAD
python -m PyInstaller --clean --noconfirm packaging\demo.spec
```

## Output Locations

| Item | Path |
| --- | --- |
| Executable | `dist/bucad-demo/bucad-demo.exe` |
| Packaging config | `packaging/demo.spec` |
| Built-in config | `dist/bucad-demo/_internal/configs/inference/demo.yml` |
| Built-in weights directory | `dist/bucad-demo/_internal/artifacts/checkpoints/` |

## Built-in Model Assets

- ConvNeXt-Tiny five-fold checkpoints: 5 files.
- EfficientNetV2-S five-fold checkpoints: 5 files.
- Segmenter checkpoint: 1 file.
- Total checkpoints: 11 files.
- Checkpoint size approximately `0.952 GB`.
- Complete release directory size approximately `4.22 GB`, mainly from PyTorch/CUDA runtime libraries and model weights.

## Smoke Test Results

- `bucad-demo.exe` can be launched from `dist/bucad-demo/` in the background.
- Local Gradio page `http://127.0.0.1:7860/` returns HTTP 200.
- Launch entry has been configured to automatically open the browser.

## Important Notes

- `dist/` and `artifacts/releases/` are not committed to the Git repository to avoid placing multi-GB binary files directly into the source code repository.
- If binary packages need to be attached to GitHub Release, release assets should be used; if a single archive exceeds GitHub's single file size limit, external cloud storage or split-volume publishing should be used instead.
- BUSI is still only used for external evaluation; the demo fixed threshold comes from the BUSBRA OOF/ROI pipeline and should not be re-tuned with BUSI at release time.
