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
- threshold sweep evidence is written to `artifacts\reports\threshold_analysis.md`

## 7. Run Handbook Model Evidence

Use the handbook main model config for the final candidate classifier:

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

Run a fast comparison smoke check before launching long experiments:

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --model-limit 1 --dry-run
```

For final report evidence, run the full comparison with the selected epoch budget:

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
```

Validation target:

- `artifacts\reports\comparison_results.json` records model names, metrics, runtime,
  fold, config path, and BUSBRA/BUSI boundary evidence
- EfficientNetV2-S is compared against the handbook baseline models on the same split

## 8. Launch the Diagnostic Demo

```powershell
python app\main.py
```

Validation target:

- the UI opens locally
- uploading one image returns probabilities, final label, and at least one
  visualization output
- if segmentation or explanation weights are absent, the UI still returns the core
  diagnosis with a visible warning

## 9. Export Visual Evidence And Batch Results

```powershell
python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence --limit 6
python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference.csv
```

Validation target:

- visual examples are saved beside `artifacts\reports\visual_evidence_review.md`
- batch CSV contains filename, probabilities, final label, status, and warnings

## 10. Package the Demo

```powershell
pyinstaller packaging\demo.spec --noconfirm
python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
```

Validation target:

- packaged output is created in `dist/`
- packaged app can access required configs and model assets from the packaged layout
- packaged run preserves the same single-image diagnostic flow as the local script run
- `artifacts\release_v1\release_v1.sha256` and
  `artifacts\reports\release_v1_manifest.md` are generated for handoff checks

## 11. Minimum Demo Checklist

- Single-image upload works
- Diagnosis result is understandable without terminal output
- Lesion overlay or explicit missing-output reason is shown
- Explanation map or explicit missing-output reason is shown
- Disclaimer text is visible on every successful diagnosis result
