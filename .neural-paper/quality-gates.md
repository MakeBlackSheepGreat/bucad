# Quality Gates

## 2026-08-20 — Stage Reviews and Closeout

- validation: PASS -> `.neural-paper/reviews/stage/validation-2026-08-20.md`.
- figure: PASS -> `.neural-paper/reviews/stage/figure-2026-08-20.md`.
- writing: PASS_WITH_NOTES resolved in artifact state: the LSENS manuscript now has eleven audited references and `.neural-paper/references.csv` matches the eleven bibliography keys. Verification record: `.neural-paper/reference-verification-2026-08-20.md`.
- polishing: PASS_WITH_NOTES resolved in artifact state: `.neural-paper/change-notes/polishing-2026-08-20.md` records the final wording, metadata, citation, and disclosure changes.

## 2026-08-20 — Professional Pre-Submission Disclosures

- G-01 supervision parity: disclosed in the Letter Discussion and `supplementary_material_lsens.tex`; the evidence-head-only control preserves architecture and deployment path but cannot equalize BBOX supervision.
- G-02 claim-protocol consistency: resolved by naming the external protocol as equal averaging of five same-architecture checkpoints. The manuscript does not describe this as a single-checkpoint result or heterogeneous fusion.
- G-03 hyperparameter sensitivity: partially addressed. Weight alternatives 0.10 to 0.30 were fold-1 screens; full five-fold sensitivity analysis remains unperformed and is disclosed as a limitation.
- G-04 evaluation-unit alignment: case-disjoint splitting is verified; classification metrics remain image level and absent case-level aggregation is disclosed.
- G-05 and G-10 novelty positioning: closely related weakly supervised localization and diagnosis work is cited. The contribution is limited to supervision placement, deployment boundary, and frozen evidence.
- G-06 annotation provenance: five fold diagnostics record BBOX validity ratio 1.0 for every validation fold; BBOX use is limited to BUSBRA training.
- G-08 metric appropriateness: fixed-threshold metrics and AUC are reported. PR curves and partial AUC are absent and are disclosed as reporting limits.
- G-09 parameter sanity: LEA has 27,823,973 parameters versus 27,821,666 for ConvNeXt-Tiny; GPU model-only latency is 8.770 ms versus 8.284 ms. Evidence: `artifacts/reports/paper_evidence/lesionext_lea_latency_benchmark.json`.
- G-11 external heterogeneity: three cohort-specific AUCs and overlapping paired bootstrap intervals are reported; no cross-sensor superiority claim is made.
- G-12 annotation-noise robustness: BBOX perturbation analysis is absent and explicitly limited in the Letter and supplementary material.
- G-13 venue alignment: the official `IEEE_lsens.cls` source compiles within the four-page Letters limit; the narrative focuses on sensor acquisition variability and deployable inference.

## 2026-08-20 — Final Local Package Verification

- Final integrity pass: `.neural-paper/final-integrity-review-2026-08-20.md` rechecked every manuscript component, citations, acknowledgments, data split, metric sources, and author-confirmation boundary. It corrected six stale standard-baseline values in the main table from the frozen OOF records.

- Innovation brief: `check_innovation_brief.py .neural-paper/innovations/LEA-20260820.md --strict` passed.
- Research discipline: `check_discipline.py . --strict` passed. Project-state validation has 0 errors; its two registry-token warnings for `ultrasound_2d` and `convnext_derived` do not affect the frozen protocol or manuscript artifacts.
- Main-text citations: 11 cited keys, 11 `bibitem` keys, 0 orphan, and 0 unused keys.
- Source and layout: main and supplementary table checks have 0 violations; main, supplementary, and cover-letter logs have 0 fatal errors and 0 overfull boxes.
- Page and rendering review: direct `pdfinfo` reports 3 pages for the Letter, 2 for the supplement, and 1 for the cover letter. All six rendered pages were visually checked for clipping, overlap, legibility, headers, footers, tables, figures, and page transitions.
- Graphical abstract: PNG and TIFF render at 3,570 by 1,470 pixels and 300 dpi.
- Numeric audit: `.neural-paper/number-drift-audit-2026-08-20.md` records the reviewed heuristic exclusions. Every outcome literal in the Letter remains traceable to frozen evidence.

## Submission Boundary

- The local package contains the complete source, PDFs, figures, graphical abstract, checklist, and current `.neural-paper` snapshot.
- Portal upload, author approval, rights, conflicts, funding confirmation, and publishing-model selection remain human confirmations. No external submission has been performed.
