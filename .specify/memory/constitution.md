<!--
Sync Impact Report
Version change: template -> 1.0.0
Modified principles:
- N/A -> I. Outcome-First Competition Scope
- N/A -> II. Reproducible Windows-First Environment
- N/A -> III. Data Boundary And Leakage Prevention
- N/A -> IV. Metric-Gated Model Development
- N/A -> V. Demo-Ready Delivery And Traceable Collaboration
Added sections:
- Technical Baseline & Repository Boundaries
- Development Workflow & Quality Gates
Removed sections:
- None
Templates requiring updates:
- UPDATED: .specify/templates/plan-template.md
- UPDATED: .specify/templates/spec-template.md
- UPDATED: .specify/templates/tasks-template.md
- UPDATED: No files found under .specify/templates/commands/; no command-template updates required
Follow-up TODOs:
- None
-->

# BUCAD Constitution

## Core Principles

### I. Outcome-First Competition Scope
All work MUST directly advance at least one competition deliverable: benign/malignant
classification, lesion segmentation, or a runnable diagnostic/demo interface. Teams
MUST prefer mature libraries, pretrained backbones, and incremental improvements over
custom architecture invention unless a comparison experiment shows a measurable
benefit. Rationale: the handbook targets a system that can run, be demonstrated, and
score within a short schedule and a novice team context.

### II. Reproducible Windows-First Environment
Development MUST target the shared Windows workflow documented in `开发手册_V2.md`;
contributors MUST use the `bucad` conda environment, Python 3.10-compatible
dependencies, and repository-documented commands. Environment, dependency, and
configuration changes MUST be recorded in versioned files or docs before teammates
rely on them. Rationale: the project depends on a five-person team reproducing the
same setup without local drift.

### III. Data Boundary And Leakage Prevention
Raw datasets and large generated artifacts MUST stay outside Git-tracked source
directories, and `.gitignore` MUST block images, masks, checkpoints, logs, and
similar bulky outputs. BUSBRA is the training dataset and MUST use case-level
splitting only; BUSI is the external evaluation dataset and MUST NOT enter training,
tuning, or validation loops. Any feature that touches data loading or evaluation MUST
include an explicit leak-prevention check. Rationale: leakage or accidental data
commits invalidates competition results and slows collaboration.

### IV. Metric-Gated Model Development
Model or preprocessing changes MUST define measurable targets and report at least the
relevant validation metrics before acceptance. For this project, AUC is the primary
score, Sensitivity is the highest-risk clinical metric, and Specificity and
supporting confusion data MUST remain inspectable. Claims about "better" models MUST
be backed by reproducible experiments, not intuition. Rationale: the handbook sets
clear thresholds (`AUC > 0.75`, target `0.85+`) and emphasizes missed malignant cases
as the key risk.

### V. Demo-Ready Delivery And Traceable Collaboration
Every milestone MUST remain runnable by another teammate through committed code,
documented commands, and artifacts placed in agreed locations such as `configs/`,
`scripts/`, `app/`, and `artifacts/`. Developers MUST commit in small logical
increments, keep README/quickstart/report guidance in sync with behavior changes, and
ensure the system degrades gracefully when optional pieces such as segmentation
weights are absent. Rationale: the project is judged not only on model quality but
also on the ability to demonstrate, package, and explain the system.

## Technical Baseline & Repository Boundaries

- Primary stack is Python 3.10, PyTorch, timm, scikit-learn, Albumentations, Gradio,
  and related scientific Python tooling described in `开发手册_V2.md`.
- The canonical repository layout for implementation planning is `configs/`, `src/`,
  `scripts/`, `app/`, `tests/`, and `artifacts/`; generated checkpoints, logs, and
  reports belong under `artifacts/`.
- Raw dataset directories such as BUSBRA and BUSI MUST live outside the Git
  repository or in ignored paths. Source control MUST store code, small configs,
  metadata, and lightweight reports only.
- Commands in onboarding docs may use local absolute Windows paths for teaching, but
  production-facing code and repeatable project scripts SHOULD accept config values
  or CLI arguments instead of requiring personal machine paths.
- GPU acceleration is preferred, but CPU fallback paths MUST remain available for
  smoke tests, onboarding, and demo recovery when CUDA is unavailable.

## Development Workflow & Quality Gates

1. Before design or implementation begins, `spec.md` or equivalent planning notes
   MUST record the affected deliverable, target dataset boundary, success metrics,
   and the exact validation command or smoke test.
2. Changes to datasets, splits, preprocessing, or evaluation MUST include a sanity
   check for sample counts, label mapping, and case-level leakage. BUSI-related work
   MUST explicitly state that BUSI is evaluation-only.
3. Changes to model training or selection MUST produce comparable evidence in
   `artifacts/reports/` or equivalent tracked documentation, including the baseline
   used and the command/config that produced the result.
4. Changes to the user-facing application, batching workflow, or packaging MUST be
   verified end to end: input image or folder, inference, output rendering/export,
   and user guidance or warning text.
5. Each logical unit of work MUST leave behind updated instructions for the next
   teammate, including required environment steps, new scripts, moved paths, or
   changed artifact names.
6. If a change cannot satisfy one of these gates, the exception MUST be written in
   the plan's Complexity Tracking or Assumptions section and approved before
   implementation continues.

## Governance

This constitution supersedes ad hoc practice notes when conflicts arise.
`开发手册_V2.md` is the operating handbook for day-to-day execution and may extend
these rules, but it MUST NOT contradict them.

Amendments MUST document the reason for change, impacted principles or sections,
affected templates/docs, and any migration work for active plans. Approval requires
review by the active project maintainers before the updated constitution is adopted.

Versioning policy follows semantic versioning for governance: MAJOR for removed or
redefined principles, MINOR for new mandatory principles or materially expanded
gates, and PATCH for clarifications that do not change required behavior.

Compliance review is mandatory for every generated `spec.md`, `plan.md`, and
`tasks.md`, and for any milestone handoff or merge review. Reviews MUST confirm
reproducible environment setup, data-boundary protection, metric evidence, and demo
readiness. Non-compliant work MUST be corrected or explicitly waived in writing
before it proceeds.

**Version**: 1.0.0 | **Ratified**: 2026-04-23 | **Last Amended**: 2026-04-23
