# Defense Q&A Preparation

Date: 2026-04-24

## Why separate BUSBRA and BUSI?

BUSBRA is used for training and internal validation, while BUSI is kept for external evaluation and demo validation. This prevents the test set from influencing model training and makes the final evaluation more credible.

## How do you prevent data leakage?

The split is case-level rather than random image-level. The generated split summary records 1,064 unique cases and `leakage_detected=false`.

## Why use EfficientNetV2-S?

The handbook selects EfficientNetV2-S as the final candidate because it is a modern, parameter-efficient CNN family suitable for limited medical-image data. The current repository has the config ready, but final superiority claims should wait until T059/T060 formal training and comparison are complete.

## What does Grad-CAM prove?

Grad-CAM does not prove clinical correctness. It provides a visual explanation of where the model focuses, which helps reviewers identify obvious mismatch or unreliable behavior.

## What are the current metric limitations?

The current BUSI final evaluation reaches AUC 0.7564, sensitivity 0.6095, specificity 0.8009, and accuracy 0.7388 at threshold 0.50. Sensitivity remains below the target, so the system should be framed as a prototype.

## Can this replace doctors?

No. The UI and defense should state that the system is for auxiliary analysis and educational/competition demonstration only.

## What happens when segmentation or explanation fails?

The inference contract supports graceful degradation: classification can still return while missing visualization reasons are shown as warnings.
## Competition Metric Completeness Note

The official competition metric set is AUC, Accuracy, Recall/Sensitivity, Precision, Specificity, and F1-Score. If an old archived table shows `-` for Precision or F1-Score, the historical summary did not preserve the confusion matrix or raw probabilities needed to reconstruct that value. For locked BUSI operating-point results, use `artifacts/reports/competition_metrics_all_busi_reports.md` as the complete metric source.

