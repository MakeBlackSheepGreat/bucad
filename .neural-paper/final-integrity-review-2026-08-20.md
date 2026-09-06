# LSENS Final Integrity Review

Date: 2026-08-20

Scope: `LesioNeXt_LEA_LSENS_Official_Submission_Package` main Letter, supplement, cover letter, graphical abstract, source figures, frozen experiment reports, configurations, split file, reference records, acknowledgments, and submission metadata.

## Content and Evidence

| Section | Status | Evidence and final disposition |
| --- | --- | --- |
| Title, abstract, keywords | Pass | Names a training-time evidence-alignment method and BBOX-free inference path. All reported outcomes match frozen records. |
| Introduction and related work | Pass | Breast-cancer burden, BUS-BRA, ConvNeXt, Swin, and three lesion-aware BUS works are cited. The acquisition-variability statement is framed as general modality context, without an unsupported sensor-robustness claim. |
| Method | Pass | `alpha=0`, alignment weight `0.25`, margin `0.08`, ConvNeXt-Tiny backbone, and equal five-checkpoint average match the classifier and inference YAML files. |
| Data and split | Pass | The validation rows in `busbra_5fold_splits.csv` contain 1,875 unique images from 1,064 cases. Each fold has 375 images and 212 or 213 cases; zero cases span validation folds. BUS-BRA metadata confirms biopsy labels and BI-RADS 2--5. |
| Statistics | Pass | Image-level pooled OOF, fixed threshold `0.50`, and paired case-cluster percentile bootstrap with 10,000 replicates and seed `20260811` match the component report. |
| Results and Table I | Corrected and pass | Six stale values inherited in an earlier table revision were replaced from the frozen OOF records: DenseNet121 Precision/F1 `0.8055`/`0.7056`; EfficientNetV2-S Specificity/Precision/F1 `0.9022`/`0.7625`/`0.7050`; Swin-Tiny Precision `0.7632`. All ranks now match the six-model source table. |
| Ablation and external results | Pass | Component metrics, evidence mass, fold AUCs, and locked external AUCs match the listed frozen JSON/Markdown reports. The text states the AUC CI crosses zero and keeps external comparisons descriptive. |
| Discussion and conclusion | Pass | Claims are bounded to evidence concentration, observed internal metrics, BBOX-free inference, and locked external reporting. It discloses supervision parity, image-level scoring, partial hyperparameter screening, and absent BBOX-noise analysis. |
| Supplement | Pass | Tables, deletion analysis, latency protocol, and limitations agree with frozen evidence. |
| Cover letter | Pass | Describes only frozen results and limitations. It has no portal directive or unsupported superiority claim. |

## References

- Citation-to-bibliography accounting: 11 cited keys, 11 `bibitem` records, zero missing, orphan, duplicate-label, or unresolved Figure/Table references.
- Crossref DOI records verified: Bray, BUS-BRA, ConvNeXt, Swin, Shin, Kim, Wang, BUSI, and Efron--Tibshirani.
- DataCite DOI records verified: BUSI-WHU and TCIA BrEaST.
- Checked fields: title, author family names, year, venue or publisher, volume/issue/page or article/dataset version, and DOI.
- BUS-BRA's Crossref metadata uses a 2023 online date and 2024 print citation; the Letter correctly cites the 2024 volume, issue, and pages.

## Acknowledgment and Author Metadata

- The acknowledgment is appropriate for an initial submission: it only thanks the creators and maintainers of BUS-BRA, BUSI, BUSI-WHU, and TCIA BrEaST.
- No anonymous-reviewer, clinical collaborator, or unsupported contribution acknowledgment remains.
- Funding wording, authorship order, affiliations, corresponding email, conflicts, ethics, data-use rights, originality, and concurrent-submission status cannot be independently proved from experiment artifacts. Their local consistency was checked, and they remain required corresponding-author confirmations in `SUBMISSION_CHECKLIST.md`.

## Package Acceptance Conditions

The corrected sources must be rebuilt, visually checked page by page for clipping or overlap, copied with the current `.neural-paper` snapshot, and packaged into a fresh ZIP. No portal upload is part of this review.
