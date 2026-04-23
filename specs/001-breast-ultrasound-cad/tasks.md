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
- [ ] T032 Run the full quickstart and final demo validation, then save evidence in `artifacts/reports/final_validation.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion - establishes the MVP
- **User Story 2 (Phase 4)**: Depends on Foundational completion and integrates with the inference service from User Story 1
- **User Story 3 (Phase 5)**: Depends on User Story 1 and consumes visualization outputs from User Story 2 when available
- **Polish (Phase 6)**: Depends on the desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: No dependency on later stories; this is the recommended MVP slice
- **User Story 2 (P2)**: Builds on the shared inference pipeline created in User Story 1
- **User Story 3 (P3)**: Requires the diagnosis pipeline from User Story 1 and should surface User Story 2 outputs or their fallback warnings

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
