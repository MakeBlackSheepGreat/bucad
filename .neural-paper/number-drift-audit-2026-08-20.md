# LSENS Number-Drift Audit

Scope: `lesionext_lea_lsens.tex` and `supplementary_material_lsens.tex` against `.neural-paper/lsens-frozen-numerics.csv` and the cited frozen OOF JSON/CSV source records.

The numerical checker intentionally treats every decimal literal as a potential metric. The CSV contains scientific results and their source reports, including the paired bootstrap limits $[-0.003, 0.032]$, $[0.005, 0.037]$, and $[-0.037, 0.027]$. It reports the absolute $0.003$ magnitude as unmatched because its literal extractor discards a leading minus sign; that value is traceable to the paired AUC lower bound. The remaining unmatched literals are manually reviewed exclusions because they are presentation or identifier values rather than outcome metrics: the two RGB color definitions and DOI fragments.

Supplement-only literals for hardware warm-up count, timed iterations, input shape, and GPU memory are source-described benchmark protocol values. They do not change the frozen outcome metrics. Missing CSV values are expected when a documented frozen result appears only in the supplementary material and do not indicate a manuscript discrepancy.

Final correction: the first closeout pass inherited six stale values in the main-table standard-baseline rows. The corrected values are DenseNet121 Precision/F1 `0.8055`/`0.7056`, EfficientNetV2-S Specificity/Precision/F1 `0.9022`/`0.7625`/`0.7050`, and Swin-Tiny Precision `0.7632`. Each now matches `all_model_comparison_v1a.csv` and its corresponding OOF JSON.

Result: every outcome literal in the Letter is traceable to the frozen numeric CSV or its cited source report; remaining checker messages are reviewed heuristic exclusions.
