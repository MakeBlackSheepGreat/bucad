# Claim-to-Evidence Map

## Evidence Map 2026-08-20 (IEEE Sensors Letters track)

### C1. Training-time evidence alignment activates the intended spatial mechanism
Claim: LesioNeXt-LEA uses BBOXs only in the training loss, keeps $\alpha=0$ for every checkpoint prediction, and concentrates evidence inside lesion boxes.
Evidence: `artifacts/reports/paper_evidence/lesionext_lea_component_ablation.json`; `artifacts/reports/paper_evidence/lesionext_lea_evidence_deletion.json`; `configs/classifier/lesionext_lens_v1a_evidence_only.yml`.
Seeds_n: 1 fixed random seed with five fold-specific retrainings; fold variation is reported and independent multi-seed repetition was not performed.
Status: supported
Boundary: BBOX alignment is training-only, although it supplies additional supervision absent from the standard image-label baselines. Evidence-mass and deletion results establish mechanism activation under the provided boxes; they do not establish robustness to BBOX annotation noise.

### C2. Frozen BUSBRA case-level five-fold development result
Claim: On 1,875 BUSBRA images from 1,064 cases with case-disjoint folds and a fixed 0.50 threshold, image-level pooled OOF metrics are AUC 0.9150, Accuracy 0.8709, Sensitivity 0.7694, Specificity 0.9196, Precision 0.8207, and F1 0.7942.
Evidence: `artifacts/reports/lesionext_lens_v1a_evidence_only_5fold_oof.json`; `artifacts/reports/paper_evidence/lesionext_lea_component_ablation.json`.
Seeds_n: 1 fixed random seed with five fold-specific retrainings; fold variation is reported and independent multi-seed repetition was not performed.
Status: supported
Boundary: The split unit is case level, while pooled classification metrics remain image level; no post-hoc per-case probability aggregation is claimed. The fixed threshold is a protocol choice, and no PR curve, partial AUC, or threshold sweep is used for the main conclusion.

### C3. Incremental comparison with ConvNeXt-Tiny is descriptive
Claim: Relative to the matched ConvNeXt-Tiny baseline, LEA changes pooled OOF AUC by +0.0144, Accuracy by +0.0208, and Sensitivity by -0.0049; the paired case-cluster AUC 95% CI is [-0.003, 0.032].
Evidence: `artifacts/reports/paper_evidence/lesionext_lea_component_ablation.json`; `artifacts/reports/fixed_classification_benchmark/oof/convnext_tiny.json`.
Seeds_n: 1 fixed random seed with five fold-specific retrainings; fold variation is reported and independent multi-seed repetition was not performed.
Status: supported
Boundary: The AUC interval includes zero, so the manuscript reports descriptive improvement and does not claim statistical superiority. The head-only control preserves architecture yet does not consume BBOX supervision, leaving a supervision-parity limitation.

### C4. Deployment boundary and checkpoint aggregation are fixed
Claim: Each deployed member is a ConvNeXt-Tiny-derived global-image classifier with identity-only preprocessing, no BBOX, crop, mask, teacher, or TTA; locked external probabilities are the equal average of five checkpoints from that one architecture.
Evidence: `configs/inference/lesionext_lens_v1a_5fold_identity.yml`; `configs/classifier/lesionext_lens_v1a_evidence_only.yml`.
Seeds_n: 1 fixed random seed with five fold-specific retrainings; fold variation is reported and independent multi-seed repetition was not performed.
Status: supported
Boundary: Five checkpoint averaging is part of the stated frozen inference protocol and is not described as a single-checkpoint result. The deployment statement excludes heterogeneous backbone fusion and localization-model inference only.

### C5. External performance is a locked descriptive robustness record
Claim: The frozen five-checkpoint identity-only protocol obtains AUC 0.9089 on BUSI, 0.8171 on BUSI-WHU, and 0.8617 on TCIA BrEaST; no external cohort selected architecture, loss, preprocessing, threshold, or checkpoint.
Evidence: `artifacts/reports/lesionext_lens_v1a_v3_external_review_en.md`; `artifacts/reports/paper_evidence/lesionext_lea_component_ablation.json`; external prediction JSON files named in the external review.
Seeds_n: 1 fixed random seed with five fold-specific retrainings; fold variation is reported and independent multi-seed repetition was not performed.
Status: supported
Boundary: Cohort-specific bootstrap intervals reflect sample uncertainty and overlap with ConvNeXt-Tiny; cross-sensor superiority is not claimed. External datasets remain locked against further model selection.

### One-sentence contribution
Training-time lesion evidence alignment concentrates learned evidence inside lesion boxes while a global-image ConvNeXt-Tiny-derived classifier remains BBOX-free at inference; its frozen five-fold protocol shows a specificity and precision tradeoff under case-disjoint development and locked external evaluation.
