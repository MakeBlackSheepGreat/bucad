# Data Model: 乳腺超声辅助诊断系统

## Entity: Dataset Source

- **Purpose**: Represent the logical role of each dataset used in the project.
- **Fields**:
  - `dataset_name`: `BUSBRA` or `BUSI`
  - `role`: `train_internal_eval` or `external_eval`
  - `root_path`: absolute or config-resolved local path
  - `label_source`: CSV metadata, folder name, or derived manifest
  - `mask_available`: boolean
- **Validation rules**:
  - `BUSBRA` MUST only map to training/internal validation flows.
  - `BUSI` MUST only map to external evaluation/demo validation flows.
  - `root_path` MUST resolve outside tracked source folders or inside ignored paths.

## Entity: Case Sample

- **Purpose**: Represent one image-centered diagnostic sample used for training,
  evaluation, or inference.
- **Fields**:
  - `sample_id`: unique image identifier
  - `case_id`: patient/case-level identifier used for leakage control
  - `dataset_name`: foreign key to Dataset Source
  - `image_path`: local image file path
  - `mask_path`: optional mask path
  - `pathology_label`: `benign`, `malignant`, or `unknown`
  - `view_side`: optional left/right marker
  - `quality_flag`: `valid`, `low_quality`, or `invalid`
- **Validation rules**:
  - `sample_id` MUST be unique within a dataset.
  - `case_id` MUST be populated for BUSBRA fold generation.
  - `pathology_label` MUST remain `unknown` for unlabeled runtime uploads.
  - `quality_flag=invalid` MUST block standard inference output.

## Entity: Split Assignment

- **Purpose**: Record which fold and stage a BUSBRA case belongs to.
- **Fields**:
  - `case_id`
  - `fold_id`
  - `stage`: `train`, `val`, or `holdout`
  - `generated_at`
  - `generator_signature`: script/config identifier
- **Validation rules**:
  - A `case_id` MUST NOT appear in both `train` and `val` for the same fold.
  - BUSI samples MUST NOT receive split assignments used by training loops.

## Entity: Training Run

- **Purpose**: Represent one classifier or segmenter experiment execution.
- **Fields**:
  - `run_id`
  - `task_type`: `classification` or `segmentation`
  - `model_name`
  - `config_path`
  - `fold_id`
  - `device`
  - `seed`
  - `status`: `planned`, `running`, `completed`, `failed`
  - `output_dir`
- **Validation rules**:
  - `config_path` MUST resolve to a versioned YAML file.
  - `seed` MUST be recorded for reproducibility.
  - `status=completed` MUST have at least one metric report and one saved artifact.

## Entity: Model Artifact

- **Purpose**: Represent a saved checkpoint or exported runtime asset.
- **Fields**:
  - `artifact_name`
  - `task_type`
  - `source_run_id`
  - `file_path`
  - `metric_summary`
  - `created_at`
  - `is_runtime_ready`
- **Validation rules**:
  - `file_path` MUST point to an ignored artifact directory.
  - `metric_summary` MUST include the metric that justified model selection.
  - `is_runtime_ready=true` requires matching config metadata and inference compatibility.

## Entity: Diagnostic Session

- **Purpose**: Represent one user-triggered inference request from the demo or a script.
- **Fields**:
  - `session_id`
  - `input_filename`
  - `submitted_at`
  - `requested_outputs`: probabilities, lesion overlay, explanation map
  - `status`: `submitted`, `validating`, `inferencing`, `rendering`, `completed`, `partial`, `failed`
  - `warning_messages`
- **Validation rules**:
  - `input_filename` MUST map to a supported image type.
  - `status=partial` is allowed only when the core diagnosis succeeds but an optional
    visualization fails.
  - `status=completed` or `partial` MUST attach a Diagnostic Result.

## Entity: Diagnostic Result

- **Purpose**: Capture the structured outcome shown to the user.
- **Fields**:
  - `session_id`
  - `benign_probability`
  - `malignant_probability`
  - `final_label`
  - `confidence_band`: `high`, `borderline`, `low`
  - `recommendation_text`
  - `analysis_timestamp`
  - `model_version`
- **Validation rules**:
  - Probabilities MUST sum to 1 within numeric tolerance.
  - `confidence_band=borderline` MUST include manual-review wording.
  - `final_label` MUST never be shown without the auxiliary-use disclaimer text.

## Entity: Visual Evidence

- **Purpose**: Hold all spatial and explanatory outputs tied to a result.
- **Fields**:
  - `session_id`
  - `overlay_type`: `lesion_mask`, `lesion_highlight`, `grad_cam`, or `none`
  - `artifact_path` or `in_memory_ref`
  - `generation_status`: `generated`, `skipped`, `failed`
  - `reason_if_missing`
- **Validation rules**:
  - At least one visual evidence record SHOULD be `generated` for successful analyses.
  - Missing optional outputs MUST include a human-readable reason.

## Relationships

- One Dataset Source has many Case Samples.
- One Case Sample may have zero or one Split Assignment in a given fold.
- One Training Run produces many Model Artifacts.
- One Diagnostic Session produces exactly one Diagnostic Result when inference reaches
  `completed` or `partial`.
- One Diagnostic Result may reference multiple Visual Evidence records.

## State Transitions

### Training Run

- `planned -> running`
- `running -> completed`
- `running -> failed`

### Diagnostic Session

- `submitted -> validating`
- `validating -> inferencing`
- `validating -> failed`
- `inferencing -> rendering`
- `inferencing -> failed`
- `rendering -> completed`
- `rendering -> partial`
- `rendering -> failed`
