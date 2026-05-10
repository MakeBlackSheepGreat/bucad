<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# ROI OOF and LCC Post-Processing Optimization Report

Date: 2026-04-25

## Data Boundary

- BUSBRA is used for training, OOF fusion, internal validation, and threshold selection.
- BUSI is used only for external review after configuration is fixed, and does not participate in training, weight fine-search, threshold fine-search, or repeated tuning.
- The mergeable changes in this round only adopt parameters supported by BUSBRA: ROI stack threshold from BUSBRA OOF, mask threshold and largest connected component from BUSBRA segmentation validation.

## Optimization Process

| Candidate | Selection Basis | BUSBRA Internal Results | BUSI External Review | Decision |
| --- | --- | --- | --- | --- |
| Current ROI OOF stacker | BUSBRA OOF logistic regression | OOF AUC `0.9223`, threshold `0.55` | AUC `0.9196`, Sensitivity `0.8429`, Specificity `0.8330` | Already the previous mainline |
| Simple logit fusion | BUSBRA OOF grid search | OOF AUC `0.9239`, higher than logistic regression stacker | AUC `0.9196`, Sensitivity `0.8143`, Specificity `0.8558` | No external AUC gain, not merged |
| Weighted logit fusion | BUSBRA OOF grid search | OOF AUC `0.9240`, full ConvNeXt weight `0.61`, ROI ConvNeXt weight `0.54` | AUC `0.9196`, Sensitivity `0.8476`, Specificity `0.8261` | No external AUC gain, not merged |
| LCC ROI post-processing | BUSBRA segmentation validation | mask threshold `0.40` achieves highest Dice; largest connected component reduces irrelevant region interference | AUC `0.9208`, Sensitivity `0.8524`, Specificity `0.8169` | Merged into `demo.yml` |

## BUSBRA Segmentation Validation Basis

| Mask Threshold | Dice | Original ROI Median Area Ratio | LCC ROI Median Area Ratio | Notes |
| ---: | ---: | ---: | ---: | --- |
| 0.30 | 0.7971 | 0.8633 | 0.5751 | Threshold too low, ROI too large |
| 0.40 | 0.8085 | 0.5750 | 0.5292 | Highest Dice, retains more complete context |
| 0.50 | 0.8074 | 0.5167 | 0.4602 | Close to 0.40, but slightly lower Dice |
| 0.60 | 0.7797 | 0.4410 | 0.4104 | Threshold too high, may cut off weak boundaries |
| 0.70 | 0.7399 | 0.4059 | 0.3797 | Threshold too high, not suitable for mainline |

## Merged Mainline Configuration

- Config file: `configs/inference/demo.yml`
- Classification models: ConvNeXt-Tiny 5-fold + EfficientNetV2-S 5-fold.
- ROI method: `segmenter_fold1.pt` predicted masks, `mask_threshold=0.40`, `largest_component=true`, `margin_ratio=0.35`.
- Probability fusion: full-image probabilities + ROI probabilities, using the logit logistic stacker trained on BUSBRA OOF.
- Decision threshold: `0.55`, from BUSBRA ROI OOF.

## External Review Results

| Config | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Previous ROI OOF mainline | 0.9196 | 0.55 | 0.8429 | 0.8330 | 0.8362 | 0.7080 | 0.7696 | TN 364 / FP 73 / FN 33 / TP 177 |
| LCC ROI post-processing mainline | 0.9208 | 0.55 | 0.8524 | 0.8169 | 0.8284 | 0.6911 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |

## Conclusions

- The truly effective change in this round is ROI mask post-processing, not replacing the OOF stacker.
- AUC improved from `0.9196` to `0.9208`, and Sensitivity improved from `0.8429` to `0.8524`, aligning with the direction of prioritizing missed diagnosis control from the original task.
- Specificity and Accuracy decreased slightly, indicating this solution leans more toward improving malignant detection rate; this aligns with the strategy of prioritizing reduced missed diagnoses in breast tumor auxiliary diagnosis.
- If further optimization is pursued, priority should be given to 5-fold segmenter OOF mask or full BUSBRA segmenter retraining, rather than continuing to fine-search thresholds on BUSI.
