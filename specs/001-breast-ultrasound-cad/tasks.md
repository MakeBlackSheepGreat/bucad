---

description: "Task list for implementing the breast ultrasound CAD prototype"
---

# Tasks: 乳腺超声辅助诊断系统

**Input**: Design documents from `/specs/001-breast-ultrasound-cad/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Validation tasks are REQUIRED. Automated smoke and integration checks are
included where practical, plus final quickstart/demo validation.

**Organization**: Tasks are grouped by user story to enable independent implementation
and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the repository skeleton, dependency manifests, and baseline config
files used by all later work.

- [X] T001 Create project scaffolding and ignore rules in `.gitignore`, `configs/`, `src/`, `scripts/`, `app/`, `tests/`, `packaging/`, and `artifacts/`
- [X] T002 Define the Python dependency manifest in `requirements.txt` and add environment smoke entry points in `check_env.py` and `check_all.py`
- [X] T003 [P] Create baseline configuration files in `configs/paths.example.yml`, `configs/classifier/baseline.yml`, `configs/segmenter/unet.yml`, and `configs/inference/demo.yml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the shared dataset, preprocessing, config, metrics, and runtime
infrastructure that blocks all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Implement BUSBRA and BUSI dataset loaders in `src/datasets/busbra.py` and `src/datasets/busi.py`
- [X] T005 Implement leakage-safe split generation and validation in `scripts/make_split.py` and `tests/smoke/test_case_split.py`
- [X] T006 [P] Implement shared config and path loading in `src/utils/config.py` and `src/utils/paths.py`
- [X] T007 [P] Implement shared image I/O and preprocessing transforms in `src/preprocess/io.py` and `src/preprocess/transforms.py`
- [X] T008 [P] Implement metrics and report writers in `src/utils/metrics.py` and `src/utils/reporting.py`
- [X] T009 Implement shared runtime, logging, and error types in `src/utils/runtime.py`, `src/utils/logging.py`, and `src/engine/errors.py`

**Checkpoint**: Foundation ready - user story implementation can now begin.

---

## Phase 3: User Story 1 - 单图诊断输出 (Priority: P1) 🎯 MVP

**Goal**: Accept one breast ultrasound image and return malignant probability,
benign probability, confidence-aware judgment, and reviewable result metadata.

**Independent Test**: Run single-image inference on a valid sample and confirm the
result includes two probabilities, final label, confidence handling, and BUSI
evaluation/report generation.

### Validation for User Story 1 (REQUIRED) ⚠️

- [X] T010 [P] [US1] Add single-image inference smoke coverage in `tests/smoke/test_single_image_inference.py`
- [X] T011 [P] [US1] Add BUSI evaluation validation in `tests/integration/test_busi_eval.py`

### Implementation for User Story 1

- [X] T012 [P] [US1] Implement the classifier model factory in `src/models/classifier.py`
- [X] T013 [P] [US1] Implement diagnostic result schema and confidence rules in `src/utils/results.py`
- [X] T014 [US1] Implement the classifier training workflow in `scripts/train_cls.py` and `src/engine/train_cls.py`
- [X] T015 [US1] Implement the single-image diagnosis service in `src/engine/inference.py`
- [X] T016 [US1] Implement BUSI evaluation and JSON report export in `scripts/eval_busi.py`
- [X] T017 [US1] Add invalid-input, low-quality, and borderline-case handling in `src/engine/inference.py` and `tests/smoke/test_single_image_inference.py`

**Checkpoint**: User Story 1 should return a complete diagnostic judgment and be
verifiable independently of UI work.

---

## Phase 4: User Story 2 - 病灶与解释可视化 (Priority: P2)

**Goal**: Produce lesion localization and explanation outputs that visually justify
the diagnostic result.

**Independent Test**: Run inference on a sample with runtime-ready weights and verify
that the outputs include a lesion visualization plus a Grad-CAM-style explanation, or
an explicit missing-output reason.

### Validation for User Story 2 (REQUIRED) ⚠️

- [X] T018 [P] [US2] Add visual evidence smoke coverage in `tests/smoke/test_visual_evidence.py`
- [X] T019 [P] [US2] Add visualization export validation in `tests/integration/test_visual_exports.py`

### Implementation for User Story 2

- [X] T020 [P] [US2] Implement the segmenter model factory in `src/models/segmenter.py`
- [X] T021 [P] [US2] Implement the Grad-CAM generator in `src/explain/gradcam.py`
- [X] T022 [US2] Implement the segmentation training workflow in `scripts/train_seg.py` and `src/engine/train_seg.py`
- [X] T023 [US2] Implement lesion overlays and explanation assembly in `src/explain/overlay.py` and `src/engine/inference.py`

**Checkpoint**: User Stories 1 and 2 together should produce diagnosis plus spatial
evidence for the same image.

---

## Phase 5: User Story 3 - 可演示的软件化流程 (Priority: P3)

**Goal**: Deliver a Gradio-based diagnostic prototype that supports upload,
analysis, result review, and packaging for demo use.

**Independent Test**: Launch the UI, upload one local image, complete the full
analysis flow, and confirm the same run can be packaged and revalidated from the
generated demo bundle.

### Validation for User Story 3 (REQUIRED) ⚠️

- [X] T024 [P] [US3] Add Gradio end-to-end smoke validation in `tests/smoke/test_gradio_flow.py`
- [X] T025 [P] [US3] Add packaged demo smoke validation in `tests/integration/test_packaged_demo.py`

### Implementation for User Story 3

- [X] T026 [P] [US3] Build the Gradio upload/result layout in `app/main.py` and `app/components/result_panels.py`
- [X] T027 [US3] Implement UI status, warning, and fallback panels in `app/components/status_panels.py` and `app/main.py`
- [X] T028 [US3] Integrate the inference service with the Gradio flow in `app/main.py` and `src/engine/inference.py`
- [X] T029 [US3] Create demo asset export and packaging support in `scripts/export_demo_assets.py` and `packaging/demo.spec`

**Checkpoint**: All three user stories should now be demonstrable through the UI and
packaging workflow.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Tighten documentation, regression coverage, and final demo readiness
across all user stories.

- [X] T030 [P] Update operator documentation in `README.md` and `specs/001-breast-ultrasound-cad/quickstart.md`
- [X] T031 [P] Add unit coverage for shared helpers in `tests/unit/test_config.py`, `tests/unit/test_metrics.py`, and `tests/unit/test_results.py`
- [X] T032 Run the full quickstart and final demo validation, then save evidence in `artifacts/reports/final_validation.md`

---

## Phase 7: Handbook Follow-up - User Story 1 Model Quality Evidence (Priority: P1)

**Source**: `开发手册_V2.md` chapters 7, 9, and 15.

**Goal**: Move beyond the runnable baseline by creating reproducible evidence for
model selection, augmentation, thresholding, and the handbook-recommended main model.

**Independent Test**: Run one smoke comparison on the same BUSBRA split and confirm
that `artifacts/reports/comparison_results.json` records AUC, Sensitivity,
Specificity, runtime, model name, config path, fold, and dataset-boundary evidence.

### Validation for Handbook Model Work (REQUIRED)

- [X] T033 [P] [US1] Add classifier backbone and comparison-config validation in `tests/unit/test_classifier_models.py` and `tests/smoke/test_comparison_config.py`
- [X] T034 [P] [US1] Add metric and threshold-analysis validation in `tests/unit/test_metrics.py` and `tests/integration/test_busi_eval.py`

### Implementation for Handbook Model Work

- [X] T035 [P] [US1] Create handbook-aligned classifier configs in `configs/classifier/efficientnetv2_s.yml` and `configs/classifier/comparison.yml`
- [X] T036 [US1] Add `basic_cnn` comparison support and EfficientNetV2-S construction coverage in `src/models/classifier.py`
- [X] T037 [US1] Add optional CLAHE and deterministic augmentation controls in `src/preprocess/transforms.py` and `configs/classifier/efficientnetv2_s.yml`
- [X] T038 [US1] Implement the handbook comparison runner in `scripts/run_comparison.py` and `src/engine/compare_cls.py`
- [X] T039 [US1] Export model-selection evidence to `artifacts/reports/comparison_results.json` and document commands in `README.md` and `specs/001-breast-ultrasound-cad/quickstart.md`
- [X] T040 [US1] Add threshold sweep output for BUSI evaluation in `src/utils/metrics.py`, `scripts/eval_busi.py`, and `artifacts/reports/threshold_analysis.md`

**Checkpoint**: User Story 1 should have reproducible proof for the selected main
classifier and threshold strategy, not only a single baseline checkpoint.

---

## Phase 8: Handbook Follow-up - User Story 2 Visual Evidence Hardening (Priority: P2)

**Source**: `开发手册_V2.md` chapters 8, 10, and 15.

**Goal**: Make lesion visualization and Grad-CAM robust enough for final demos,
including the handbook's recommended EfficientNetV2-S backbone and clinical review
evidence.

**Independent Test**: Export a visual evidence pack for selected BUSI/BUSBRA samples
and confirm every sample has either lesion overlay plus Grad-CAM or a clear missing
reason.

### Validation for Visual Evidence Work (REQUIRED)

- [X] T041 [P] [US2] Add Grad-CAM target-layer resolution tests for ResNet-style and EfficientNetV2-style backbones in `tests/unit/test_gradcam_targets.py`
- [X] T042 [P] [US2] Add visual-evidence export validation in `tests/integration/test_visual_exports.py`

### Implementation for Visual Evidence Work

- [X] T043 [US2] Update Grad-CAM target-layer resolution for `layer4`, `features`, `blocks`, `conv_head`, and supported timm backbones in `src/models/classifier.py`
- [X] T044 [US2] Add visual evidence export script for report-ready overlays in `scripts/export_visual_evidence.py`
- [X] T045 [US2] Save representative overlay and heatmap review notes in `artifacts/reports/visual_evidence_review.md`
- [X] T046 [US2] Document visual-evidence review commands and missing-output behavior in `specs/001-breast-ultrasound-cad/quickstart.md`

**Checkpoint**: User Story 2 should be explainable with report-ready visual evidence,
and Grad-CAM should not silently break when the classifier backbone changes.

---

## Phase 9: Handbook Follow-up - User Story 3 Release And Defense Readiness (Priority: P3)

**Source**: `开发手册_V2.md` chapters 11, 12, and 15.

**Goal**: Turn the prototype into a final handoff package with frozen runtime
weights, batch/demo workflows, checksums, report tables, and rehearsal evidence.

**Independent Test**: Build `artifacts/release_v1/`, launch the packaged demo from
the exported release assets, and verify a single-image result plus release checksum
manifest.

### Validation for Release Work (REQUIRED)

- [X] T047 [P] [US3] Add release-manifest validation in `tests/integration/test_packaged_demo.py`
- [X] T048 [P] [US3] Add batch-inference smoke validation in `tests/smoke/test_batch_inference.py`

### Implementation for Release Work

- [X] T049 [US3] Add release asset export with SHA-256 manifest in `scripts/export_demo_assets.py` and `artifacts/reports/release_v1_manifest.md`
- [X] T050 [US3] Add optional 5-fold classifier ensemble runtime support in `src/engine/inference.py` and `configs/inference/demo.yml`
- [X] T051 [US3] Implement batch inference export in `scripts/batch_infer.py` and expose the validated command in `README.md`
- [X] T052 [US3] Update packaging guidance for `artifacts/release_v1/` in `packaging/demo.spec`, `README.md`, and `specs/001-breast-ultrasound-cad/quickstart.md`
- [X] T053 [US3] Record demo rehearsal and defense checklist evidence in `artifacts/reports/demo_rehearsal.md`

**Checkpoint**: User Story 3 should be ready for final presentation, packaging,
report writing, and teammate handoff.

---

## Phase 10: Final Handbook Evidence Freeze

**Purpose**: Lock the final experiment evidence and keep implementation, docs, and
release artifacts synchronized before the final delivery window.

- [X] T054 [P] Generate report-ready model tables from comparison, BUSI evaluation, and threshold reports in `artifacts/reports/report_tables.md`
- [X] T055 [P] Update final validation evidence after handbook follow-up work in `artifacts/reports/final_validation.md`
- [X] T056 Run full regression plus selected handbook smoke commands, then update completion status in `specs/001-breast-ultrasound-cad/tasks.md`

---

## Phase 11: Handbook Formal Experiments - Final Model Evidence (Priority: P1)

**Source**: `开发手册_V2.md` chapters 5, 7, 9, 12, and 15.

**Goal**: Convert the current validated pipeline into formal experiment evidence that
can support final model-selection claims in the report and defense.

**Independent Test**: Confirm `artifacts/reports/model_freeze_decision.md` cites
completed EfficientNetV2-S fold metrics, full comparison results, BUSI external
evaluation, threshold choice, and exact commands/configs used.

### Validation for Formal Experiment Work (REQUIRED)

- [ ] T057 [P] [US1] Verify formal experiment prerequisites and record dataset/checkpoint availability in `artifacts/reports/formal_experiment_prerequisites.md`
- [ ] T058 [P] [US1] Add a formal-run command checklist for 5-fold training and comparison in `artifacts/reports/formal_experiment_commands.md`

### Implementation for Formal Experiment Work

- [ ] T059 [US1] Run EfficientNetV2-S 5-fold training and save metrics/checkpoints under `artifacts/reports/train_cls_efficientnetv2_s_fold{fold}.json` and `artifacts/checkpoints/efficientnetv2_s_fold{fold}.pt`
- [ ] T060 [US1] Run the full handbook comparison with `configs/classifier/comparison.yml` and update `artifacts/reports/comparison_results.json`
- [ ] T061 [US1] Run BUSI evaluation for the frozen final classifier or ensemble and save `artifacts/reports/busi_eval_final.json`
- [ ] T062 [US1] Write the final model and threshold freeze decision in `artifacts/reports/model_freeze_decision.md`

**Checkpoint**: User Story 1 should have real long-run evidence, not only smoke
validation, before claiming EfficientNetV2-S is the best model.

---

## Phase 12: Handbook Formal Visual Review - Explainability Evidence (Priority: P2)

**Source**: `开发手册_V2.md` chapters 8, 10, and 15.

**Goal**: Produce final visual examples and human review notes that show lesion
localization and Grad-CAM outputs are understandable and not misleading.

**Independent Test**: Confirm `artifacts/reports/visual_evidence_review.md` lists
reviewed benign and malignant cases, expected lesion area, Grad-CAM plausibility,
and any failure/missing-output reason.

### Validation for Formal Visual Review (REQUIRED)

- [ ] T063 [P] [US2] Export final benign/malignant visual examples into `artifacts/reports/visual_evidence_final/README.md`
- [ ] T064 [P] [US2] Record segmentation and Grad-CAM plausibility review notes in `artifacts/reports/visual_evidence_review.md`

### Implementation for Formal Visual Review

- [ ] T065 [US2] Select final representative BUSI/BUSBRA cases and document sample IDs in `artifacts/reports/final_visual_cases.md`
- [ ] T066 [US2] Capture report-ready original, lesion overlay, and heatmap figure index in `artifacts/reports/final_figures.md`

**Checkpoint**: User Story 2 should have final visual evidence ready for report
screenshots and defense explanation.

---

## Phase 13: Handbook Release Candidate Freeze (Priority: P3)

**Source**: `开发手册_V2.md` chapters 11, 12, and 15.

**Goal**: Freeze the runnable release candidate with final configs, final checkpoints,
checksums, batch export, and packaged-demo validation.

**Independent Test**: Launch the packaged demo from the frozen release assets and
record the result in `artifacts/reports/final_packaged_demo.md`.

### Validation for Release Candidate Freeze (REQUIRED)

- [ ] T067 [P] [US3] Validate final runtime config and checkpoint references in `configs/inference/demo.yml` and `artifacts/reports/release_v1_manifest.md`
- [ ] T068 [P] [US3] Run final batch inference export and save `artifacts/reports/batch_inference_final.csv`

### Implementation for Release Candidate Freeze

- [ ] T069 [US3] Export final release assets and SHA-256 manifest into `artifacts/release_v1/` and `artifacts/reports/release_v1_manifest.md`
- [ ] T070 [US3] Run packaged demo smoke from final release assets and record evidence in `artifacts/reports/final_packaged_demo.md`
- [ ] T071 [US3] Complete the live Chinese UI rehearsal checklist in `artifacts/reports/demo_rehearsal.md`

**Checkpoint**: User Story 3 should be frozen as a repeatable demo package suitable
for handoff, presentation, and offline recovery.

---

## Phase 14: Handbook Report And Defense Package

**Source**: `开发手册_V2.md` chapters 9, 12, and 15.

**Purpose**: Turn the final experiment outputs into report tables, presentation
talking points, and final handoff notes.

- [ ] T072 [P] Convert final comparison, BUSI, threshold, and segmentation outputs into report-ready tables in `artifacts/reports/report_tables.md`
- [ ] T073 [P] Draft report/PPT outline and defense talking points in `artifacts/reports/defense_outline.md`
- [ ] T074 [P] Prepare likely defense Q&A based on model choice, data leakage, metrics, Grad-CAM, and limitations in `artifacts/reports/defense_qa.md`
- [ ] T075 Create final handoff checklist with commands, artifacts, risks, and owners in `artifacts/reports/final_handoff.md`
- [ ] T076 Run final full validation after release freeze and update `artifacts/reports/final_validation.md` and `specs/001-breast-ultrasound-cad/tasks.md`
- [X] T077 [P] Add readable DOCX/PDF report export support in `src/utils/document_reports.py` and `scripts/export_report_documents.py`
- [X] T078 [P] Document and validate report document export in `README.md`, `specs/001-breast-ultrasound-cad/quickstart.md`, and `tests/integration/test_report_documents.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion - establishes the MVP
- **User Story 2 (Phase 4)**: Depends on Foundational completion and integrates with the inference service from User Story 1
- **User Story 3 (Phase 5)**: Depends on User Story 1 and consumes visualization outputs from User Story 2 when available
- **Polish (Phase 6)**: Depends on the desired user stories being complete
- **Handbook Model Quality (Phase 7)**: Depends on Phase 6 completion and should complete before final model freeze
- **Handbook Visual Evidence (Phase 8)**: Depends on Phase 7 model decisions for backbone-specific Grad-CAM checks
- **Handbook Release Readiness (Phase 9)**: Depends on Phase 7 runtime weights and Phase 8 visual evidence behavior
- **Final Handbook Evidence Freeze (Phase 10)**: Depends on Phases 7-9 completion
- **Formal Model Evidence (Phase 11)**: Depends on Phase 10 and requires GPU/runtime availability for long experiments
- **Formal Visual Review (Phase 12)**: Depends on Phase 11 frozen or candidate weights
- **Release Candidate Freeze (Phase 13)**: Depends on Phase 11 final weights and Phase 12 final visual evidence behavior
- **Report And Defense Package (Phase 14)**: Depends on Phases 11-13 final evidence

### User Story Dependencies

- **User Story 1 (P1)**: No dependency on later stories; this is the recommended MVP slice
- **User Story 2 (P2)**: Builds on the shared inference pipeline created in User Story 1
- **User Story 3 (P3)**: Requires the diagnosis pipeline from User Story 1 and should surface User Story 2 outputs or their fallback warnings
- **Handbook follow-up US1**: Prioritize first because model-selection evidence drives report claims and final runtime weights
- **Handbook follow-up US2**: Run after the selected classifier backbone is stable so Grad-CAM targets match the real model
- **Handbook follow-up US3**: Run after model and visual evidence are frozen so release assets are reproducible
- **Formal evidence US1**: Must complete before report claims about EfficientNetV2-S superiority
- **Formal evidence US2**: Must complete before final screenshots and explainability defense
- **Formal evidence US3**: Must complete before final packaged demo and handoff

### Within Each User Story

- Validation tasks come first
- Model factories and schemas precede orchestration
- Training/evaluation scripts precede demo-level integration
- Story-specific error handling is completed before the story checkpoint

### Parallel Opportunities

- `T003` can run in parallel with `T002` after the directory skeleton exists
- `T006`, `T007`, and `T008` can run in parallel once dataset file targets are fixed by `T004`
- `T010` and `T011` can run in parallel before `T015` and `T016`
- `T020` and `T021` can run in parallel before `T022` and `T023`
- `T024` and `T025` can run in parallel before `T028` and `T029`
- `T030` and `T031` can run in parallel during the polish phase
- `T033` and `T034` can run in parallel before model-quality implementation tasks
- `T041` and `T042` can run in parallel before visual-evidence hardening tasks
- `T047` and `T048` can run in parallel before release implementation tasks
- `T054` and `T055` can run in parallel once Phases 7-9 have produced evidence
- `T057` and `T058` can run in parallel before long formal experiments
- `T063` and `T064` can run in parallel once final checkpoints are available
- `T067` and `T068` can run in parallel before final release export
- `T072`, `T073`, and `T074` can run in parallel after final metrics and visuals are frozen

---

## Parallel Example: User Story 1

```bash
# Validation work that can run in parallel
Task: "Add single-image inference smoke coverage in tests/smoke/test_single_image_inference.py"
Task: "Add BUSI evaluation validation in tests/integration/test_busi_eval.py"

# Model/schema work that can run in parallel
Task: "Implement the classifier model factory in src/models/classifier.py"
Task: "Implement diagnostic result schema and confidence rules in src/utils/results.py"
```

## Parallel Example: User Story 2

```bash
Task: "Implement the segmenter model factory in src/models/segmenter.py"
Task: "Implement the Grad-CAM generator in src/explain/gradcam.py"
```

## Parallel Example: User Story 3

```bash
Task: "Add Gradio end-to-end smoke validation in tests/smoke/test_gradio_flow.py"
Task: "Add packaged demo smoke validation in tests/integration/test_packaged_demo.py"
```

## Parallel Example: Handbook Follow-up

```bash
Task: "Add classifier backbone and comparison-config validation in tests/unit/test_classifier_models.py and tests/smoke/test_comparison_config.py"
Task: "Add metric and threshold-analysis validation in tests/unit/test_metrics.py and tests/integration/test_busi_eval.py"
Task: "Add Grad-CAM target-layer resolution tests in tests/unit/test_gradcam_targets.py"
```

## Parallel Example: Formal Delivery Work

```bash
Task: "Verify formal experiment prerequisites and record dataset/checkpoint availability in artifacts/reports/formal_experiment_prerequisites.md"
Task: "Add a formal-run command checklist for 5-fold training and comparison in artifacts/reports/formal_experiment_commands.md"
Task: "Draft report/PPT outline and defense talking points in artifacts/reports/defense_outline.md"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Stop and validate single-image diagnosis plus BUSI report export
5. Use this as the first demoable milestone

### Incremental Delivery

1. Finish Setup and Foundational once
2. Deliver User Story 1 as the baseline diagnosis pipeline
3. Add User Story 2 to enrich results with lesion/explanation evidence
4. Add User Story 3 to expose the flow through the UI and packaging
5. Finish with Phase 6 for regression coverage and final evidence capture
6. Continue with the handbook follow-up phases: model evidence, visual evidence hardening, release readiness, and final evidence freeze
7. Finish with formal handbook delivery phases: long-run experiments, visual review, release candidate freeze, report/defense package

### Parallel Team Strategy

1. One developer handles dataset/config foundations
2. One developer focuses on classifier/inference tasks in User Story 1
3. One developer can prepare Grad-CAM and segmenter components for User Story 2 once foundations are stable
4. UI and packaging tasks start after User Story 1 is usable and incorporate User Story 2 outputs as they land

---

## Notes

- All tasks follow the required checklist format with IDs, labels, and file paths
- Validation/report tasks are included for every user story
- User Story 1 is the recommended MVP scope
- User Story 3 is intentionally scheduled after core model work to preserve demo stability
- The next executable scope is Phase 11 because the implementation and smoke evidence are complete, while formal long-run evidence and defense assets remain open
