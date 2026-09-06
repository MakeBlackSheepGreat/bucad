# Decision Log


## 2026-08-20 — Target venue: IEEE Sensors Journal (non-OA traditional)
- Decision: Migrate from TBME to IEEE Sensors Journal, traditional subscription track (non-OA), per user request via neural-paper-workflow. Sensors scope reframing: breast ultrasound as sensor signal.
- Rationale: SPL requires algorithmic signal-processing contribution; TBME demands stronger clinical significance. Sensors is most permissive IEEE non-OA venue for deployed sensor-robust CAD. Existing LEA evidence supports sensor-aware regularization narrative: training-time BBOX alignment, deployment global inference, frozen case-level 5-fold and 3 locked external cohorts.
- Impact: Requires IEEE Sensors Journal LaTeX template, reframe abstract/methods for sensor acquisition variability, keep frozen numbers and case-level protocol. No model retraining required.
- BOOTSTRAP: manifest.target_venue = ieee_sensors_journal

## 2026-08-20 — IEEE Sensors Journal draft complete (neural-paper-workflow writing)
- What was done: Migrated TBME manuscript to IEEE Sensors Journal using IEEEtran journal class (traditional non-OA track). Added sensor-aware framing (acquisition sensor chain, cross-sensor robustness, deployment without localization). Preserved all frozen numbers, case-level 5-fold protocol, and locked externals. Compiled main PDF (5 pages) and supplementary (1 page) with IEEEtran.
- Verification: latexmk success, 5 pages (limit 8), PDF 363kB, figures included. Supplementary compiled 1 page. No retraining, no external leakage.
- Locations:
  - Main: C:/Users/Sakura/Desktop/BlackSheep/Paper/LesioNeXt_LEA_IEEE_Sensors/lesionext_lea_sensors.tex / .pdf
  - Supplementary: supplementary_material_sensors.tex / .pdf
  - Figures: figures/lesionext_lea_architecture.pdf, evidence_alignment_quantification.pdf
- Remaining: Submission package assembly, cover letter, final page/figure/ref checks before upload.
## 2026-08-20 — Target venue switch to IEEE Sensors Letters (non-OA traditional, ESCI Q3) per user insistence
- Decision: Switch from ieee_sensors_journal to ieee_sensors_letters, traditional subscription track (non-OA), despite ESCI Q3 status (JIF 2.2) vs Journal SCIE Q1 (JIF 4.3). User prioritizes Q3/4 ease and 4-page Letters format.
- Rationale: User explicitly requests Letters despite earlier recommendation for Journal ease on page limit; Letters 4pp+1ref hard limit requires extreme compression and stronger sensor-novelty framing. No new training; only manuscript restructuring, extreme polishing and optimization within Letters constraints.
- Impact: Existing 5pp Journal draft must be restructured to 4pp Letters IEEEtran, tables/figures compressed, supplementary separated (if allowed), cover letter and graphical abstract updated to Letters requirements. Frozen evidence unchanged.
## 2026-08-20 — IEEE Sensors Letters official template reconstruction + extreme polish and optimization (neural-paper-workflow)
- What was done: Reconstructed Letters manuscript on official IEEE_lsens.cls V1.0 2017/07/12 (C:\Users\Sakura\Desktop\BlackSheep\Paper\IEEE_lsens\bare_lsens.tex) with \IEEEtitleabstractindextext, \IEEELSENSarticlesubject, \IEEELSENSmanuscriptreceived, \IEEEpubid, \IEEEpubidadjcol. Content extreme-polished: sensor chain framing tightened, abstract compressed without changing frozen numbers (AUC 0.9150 etc), Methods deployment boundary and alignment loss clarified, Results pooled primary with paired case-cluster CI, component ablation single-factor, locked externals. Table booktabs scriptsize 8 cols, figures single-column with trim/clip and panel (a)(b) inside axes. Compiled to 3 pages (limit 4), 355509 bytes, letter paper, pdfLaTeX TeXLive 2026.
- Verification: latexmk success, 0 errors, Underfull badness 1352/1552 only, pdfinfo 3 pages, validate_project_state 0 errors 2 warnings. Numbers verified against lesionext_lens_v1a_evidence_only_5fold_oof.json and fixed_classification_benchmark.
- Locations:
  - Official draft: C:\Users\Sakura\Desktop\BlackSheep\Paper\LesioNeXt_LEA_LSENS_Official\lesionext_lea_lsens.tex / .pdf
  - Package: C:\Users\Sakura\Desktop\BlackSheep\Paper\LesioNeXt_LEA_LSENS_Official_Submission_Package\ + .zip (578072 bytes) including IEEE_lsens.cls and figures
  - Evidence map: .neural-paper/evidence-map.md updated to Letters track
- Remaining: Cover letter and graphical abstract update to Letters 4-point requirements, final reference verification, then submission via IEEE Author Portal (Main Manuscript zip, Graphical Abstract, Cover Letter separate).

## 2026-08-20 — LSENS pre-submission integrity closeout

- Decision: Retain the frozen LesioNeXt-LEA experiment set; no retraining is justified by the current evidence audit. Finish the submission package through documentation, consistency, and disclosure corrections.
- Evidence: The five required v1a checkpoints exist; the frozen BUSBRA pooled OOF, component ablation, evidence deletion, latency benchmark, and locked external reports are present. The main technical risk found in closeout was textual: calling five same-architecture checkpoint averaging a single-model result obscures the deployment protocol.
- Changes: The LSENS source now states equal averaging of five fold-specific checkpoints of one ConvNeXt-Tiny-derived architecture. Publication-stage template fields were removed. Related-work and dataset citations were audited. The supplementary material records G-01, G-03, G-04, G-08, G-09, G-11, and G-12 limits. A graphical abstract, cover-letter draft, and author-confirmation checklist were added.
- Red-flag self-check: Motivation check: the positive AUC and specificity deltas made concise deployment wording attractive. Alternative explanation: gains may arise from additional BBOX supervision, fold averaging, or the fixed decision threshold rather than an independently established sensor-robust mechanism. Falsification: the no-alignment and local-interpolation controls, paired case-cluster intervals, evidence deletion, and explicit limitations restrict the conclusion; a future BBOX perturbation, case-level aggregation, and multi-seed study could change the interpretation.
- Submission boundary: The output is a locally built package. The corresponding author must confirm authorship, originality, rights, conflicts, funding, publication model, and current Author Portal requirements before upload. No portal action is authorized or performed.

## 2026-08-20 — Final claim-scope correction

- Decision: Supersede earlier drafting language that described LEA as sensor-robust or as demonstrating cross-sensor robustness.
- Rationale: The frozen record establishes case-disjoint internal evaluation and locked cohort-specific external results. It does not identify acquisition devices per sample or test a causal sensor-robustness mechanism.
- Impact: The final Letter uses acquisition variability only as application context and reports the three locked external AUCs descriptively. The submission snapshot manifest now identifies locked external-cohort evaluation, and the source citation for BUS-BRA is limited to the dataset and protocol statement.
