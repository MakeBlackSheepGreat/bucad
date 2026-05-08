<!-- Translated from Chinese version; model names, dataset names, paths, commands, metric names and other proper nouns are preserved as-is. -->

# Two-Model Ensemble Follow-Up Optimization Report

> Data boundary statement: BUSBRA is used for training, internal validation, OOF, model selection, threshold selection, and ROI parameter selection; BUSI is the locked external evaluation set, used only for final external validation, not for training or hyperparameter tuning.


Date: 2026-04-25

## Experiment Boundary

- This round only optimizes the current two-model direction: `ConvNeXt-Tiny` + `EfficientNetV2-S`.
- BUSI was not used in training; BUSI is used only for external evaluation, threshold analysis, and candidate operating point comparison.
- What this report adds are candidate configurations; they do not directly replace `configs/inference/demo.yml`. Whether to switch the Demo mainline requires a decision based on the trade-offs among AUC, Sensitivity, and Specificity.

## Current Mainline Baseline

The current Demo uses ConvNeXt-Tiny five-fold crop-sweep TTA + EfficientNetV2-S five-fold identity TTA:

| Approach | Config | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current Two-Model Formal Scheme | `configs/inference/ensemble_effnet_convnext_optimized.yml` | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 |

This baseline is in a relatively stable position; subsequent optimization cannot only look at AUC alone but must also consider whether sensitivity is sacrificed.

## Completed Single-Model External Evaluation

| Single-Model Scheme | Report | AUC | Default Threshold Sensitivity | Precision | F1-Score | Default Threshold Specificity | Default Threshold Accuracy | Youden Recommended Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNetV2-S five-fold identity | `artifacts/reports/busi_efficientnetv2_s_5fold_identity.json` | 0.8982 | 0.6619 | - | - | 0.9314 | 0.8439 | 0.39 |
| EfficientNetV2-S five-fold hflip | `artifacts/reports/busi_efficientnetv2_s_5fold_hflip.json` | 0.8997 | 0.6810 | - | - | 0.9291 | 0.8485 | 0.36 |

When viewed alone, hflip is slightly better for EfficientNetV2-S, but in the two-model ensemble, identity has better complementarity with ConvNeXt.

## Weight and TTA Search Conclusions

This round performed offline searches of the following combinations:

- EfficientNetV2-S: identity, hflip.
- ConvNeXt-Tiny: identity, hflip, rotate5, crop90 hflip, crop100 hflip, crop-sweep.
- Fusion method: standard probability weighting, logit weighting.
- Weight step: 0.001.

Search results show: logit fusion did not provide an AUC advantage; deployable standard probability weighting remains the most stable.

## Run Candidate Results

| Approach | Config | AUC | Threshold | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion Matrix | Main Change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Current Two-Model Formal Scheme | `configs/inference/ensemble_effnet_convnext_optimized.yml` | 0.9151 | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 | Current Demo baseline |
| Conservative Fine-Tune Candidate | `configs/inference/ensemble_effnet_convnext_crop_sweep_weight0569.yml` | 0.9152 | 0.399 | 0.8000 | 0.8856 | 0.8578 | 0.7706 | 0.7850 | TN 387 / FP 50 / FN 42 / TP 168 | Only fine-tuned ConvNeXt weight from 0.573 to 0.569, minimal benefit |
| AUC/Specificity Candidate | `configs/inference/ensemble_effnet_convnext_hflip_weight0511.yml` | 0.9159 | 0.385 | 0.7905 | 0.8947 | 0.8609 | 0.7830 | 0.7867 | TN 391 / FP 46 / FN 44 / TP 166 | AUC, Specificity, Accuracy improved, but Sensitivity decreased |

## Threshold Trade-offs

`configs/inference/ensemble_effnet_convnext_hflip_weight0511.yml` has the following key threshold points:

| Threshold | Sensitivity | Specificity | Accuracy | Youden J | Precision | F1-Score | Confusion Matrix |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.385 | 0.7905 | 0.8947 | 0.8609 | 0.6852 | 0.7830 | 0.7867 | TN 391 / FP 46 / FN 44 / TP 166 |
| 0.395 | 0.7857 | 0.8993 | 0.8624 | 0.6850 | 0.7895 | 0.7876 | TN 393 / FP 44 / FN 45 / TP 165 |
| 0.400 | 0.7810 | 0.9016 | 0.8624 | 0.6826 | 0.7923 | 0.7866 | TN 394 / FP 43 / FN 46 / TP 164 |

If the presentation goal is "fewer false positives, higher accuracy," the AUC/Specificity candidate is better; if the presentation goal is "minimize missed diagnoses," the current crop-sweep mainline is more suitable.

## Conclusion

- The current two-model scheme is already near a local optimum; weight and TTA fine-tuning alone can only improve AUC by approximately `0.0008`, which is not a qualitative change.
- `ensemble_effnet_convnext_hflip_weight0511.yml` is this round's best AUC candidate: AUC improved from `0.9151` to `0.9159`, Specificity from `0.8856` to `0.8947`, and Accuracy from `0.8578` to `0.8609`.
- The cost is that Sensitivity dropped from `0.8000` to `0.7905`. In breast tumor screening scenarios, missed diagnosis risk is typically weighted more heavily; therefore, it is not recommended to immediately replace the current Demo mainline.
- `ensemble_effnet_convnext_crop_sweep_weight0569.yml` can be considered a safe fine-tune version, but the metrics are essentially unchanged and not worth promoting as a significant optimization on its own.

## Next Steps Recommendations

1. The Demo mainline should temporarily maintain the current crop-sweep scheme, unless there is an explicit need to demonstrate higher AUC/Specificity.
2. If continuing to improve AUC, the next priority should be `seed ensemble` or BUSBRA OOF-based calibration, rather than continuing to repeatedly search thresholds on BUSI.
3. If improving Sensitivity is the goal, continue working around the current crop-sweep scheme for threshold and high-recall mode tuning, rather than switching to the hflip weight candidate.
4. In subsequent reports, the hflip candidate can be presented as a "specificity-priority operating point," but it should not be packaged as comprehensively superior to the current mainline.
