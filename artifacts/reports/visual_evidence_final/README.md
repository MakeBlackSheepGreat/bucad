# Final Visual Evidence

Date: 2026-04-24

Export command:

`conda run -n BUCAD python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence_final --limit 6`

## Case Index

| Case | Original | Lesion Overlay | Grad-CAM Explanation | Status |
| --- | --- | --- | --- | --- |
| `001_benign_(1)` | `001_benign_(1)/original.png` | `001_benign_(1)/lesion_overlay.png` | `001_benign_(1)/explanation.png` | complete |
| `002_benign_(10)` | `002_benign_(10)/original.png` | `002_benign_(10)/lesion_overlay.png` | `002_benign_(10)/explanation.png` | complete |
| `003_benign_(100)` | `003_benign_(100)/original.png` | `003_benign_(100)/lesion_overlay.png` | `003_benign_(100)/explanation.png` | complete |
| `004_benign_(101)` | `004_benign_(101)/original.png` | `004_benign_(101)/lesion_overlay.png` | `004_benign_(101)/explanation.png` | complete |
| `005_benign_(102)` | `005_benign_(102)/original.png` | `005_benign_(102)/lesion_overlay.png` | `005_benign_(102)/explanation.png` | complete |
| `006_benign_(103)` | `006_benign_(103)/original.png` | `006_benign_(103)/lesion_overlay.png` | `006_benign_(103)/explanation.png` | complete |

## Interpretation Notes

- These examples are report-ready visual evidence exports from the current frozen inference configuration.
- The current export is benign-heavy because the exporter walks the BUSI folders in deterministic order; malignant batch evidence is recorded separately in `artifacts/reports/batch_inference_final.csv`.
- Visual overlays are for auxiliary explanation only and must not be presented as clinical ground truth.
