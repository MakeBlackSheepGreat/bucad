# Quickstart: 乳腺超声辅助诊断系统

## 1. Prerequisites

- Windows 10/11
- Conda available on `PATH`
- Python 3.10
- Optional NVIDIA GPU with working CUDA runtime
- Local BUSBRA and BUSI datasets stored outside the Git-tracked source tree

## 2. Environment Setup

```powershell
conda create -n bucad python=3.10 -y
conda activate bucad
pip install -r requirements.txt
python check_env.py
python check_all.py
```

Expected result:

- `check_env.py` confirms PyTorch import success
- CUDA is reported when a supported GPU is present
- CPU fallback remains usable when CUDA is unavailable

## 3. Configure Local Paths

Create a local path configuration file from the example template:

```powershell
Copy-Item configs\paths.example.yml configs\paths.local.yml
```

Expected path keys:

- `busbra_root`
- `busi_root`
- `artifacts_root`
- `default_classifier_ckpt`
- `default_segmenter_ckpt`

## 4. Generate or Verify Split Files

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

Validation target:

- BUSBRA split generation succeeds
- case-level overlap check reports zero leakage

## 5. Smoke-Train the Classifier

```powershell
python scripts\train_cls.py --config configs\classifier\baseline.yml --fold 1 --epochs 1
```

Validation target:

- training loop starts
- one validation pass completes
- checkpoint and metric report are written under `artifacts/`

## 6. Run External Evaluation

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml
```

Validation target:

- BUSI is loaded as evaluation-only data
- AUC, Sensitivity, Specificity, and confusion values are written to
  `artifacts\reports\`

## 7. Launch the Diagnostic Demo

```powershell
python app\main.py
```

Validation target:

- the UI opens locally
- uploading one image returns probabilities, final label, and at least one
  visualization output
- if segmentation or explanation weights are absent, the UI still returns the core
  diagnosis with a visible warning

## 8. Package the Demo

```powershell
pyinstaller packaging\demo.spec --noconfirm
python scripts\export_demo_assets.py --config configs\paths.local.yml
```

Validation target:

- packaged output is created in `dist/`
- packaged app can access required configs and model assets from the packaged layout
- packaged run preserves the same single-image diagnostic flow as the local script run

## 9. Minimum Demo Checklist

- Single-image upload works
- Diagnosis result is understandable without terminal output
- Lesion overlay or explicit missing-output reason is shown
- Explanation map or explicit missing-output reason is shown
- Disclaimer text is visible on every successful diagnosis result
