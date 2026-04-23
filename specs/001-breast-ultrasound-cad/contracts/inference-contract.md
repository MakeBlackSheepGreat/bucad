# Inference Contract: Single Image Diagnosis

## Purpose

Define the user-facing and script-facing contract for one breast ultrasound image
diagnosis request.

## Request Contract

- **Entry points**:
  - Gradio upload action in `app/main.py`
  - Future script wrapper for local smoke inference
- **Required inputs**:
  - `image`: one local image uploaded by the user
- **Optional inputs**:
  - `case_id`: optional text identifier for exported reports
  - `decision_threshold`: optional override for the malignant/benign decision boundary
  - `need_segmentation`: default `true`
  - `need_explanation`: default `true`

## Input Validation Rules

- Accept common static image files used in the project workflow, such as PNG, JPG, or JPEG.
- Reject empty, corrupted, or unreadable files before inference starts.
- Reject or warn on images that cannot be normalized into the expected single-image
  inference format.
- If segmentation or explanation weights are absent, continue only if the core
  classification path is healthy.

## Response Contract

- **Core outputs**:
  - `benign_probability`
  - `malignant_probability`
  - `final_label`
  - `confidence_band`
  - `recommendation_text`
  - `auxiliary_use_disclaimer`
- **Visual outputs**:
  - `original_image_view`
  - `lesion_overlay_view` or `lesion_visualization_missing_reason`
  - `explanation_view` or `explanation_missing_reason`
- **Metadata**:
  - `input_filename`
  - `analysis_timestamp`
  - `model_identifier`
  - `warnings`

## Success Conditions

- Core outputs must always be present for a successful diagnosis.
- At least one visual output should be present for a fully successful diagnosis.
- If optional outputs fail, the response status must downgrade to `partial` and expose
  a clear warning rather than silently omitting the content.

## Failure Contract

- `invalid_input`: unsupported or unreadable image
- `quality_blocked`: image quality too poor for reliable analysis
- `classification_unavailable`: no runtime-ready classifier available
- `unexpected_runtime_error`: uncaught processing error

Each failure must return:

- a user-readable message
- whether retry is reasonable
- whether manual review is recommended

## Example Review Layout

1. Original ultrasound image
2. Probability and final judgment block
3. Lesion localization overlay block
4. Explanation heatmap block
5. Warning/disclaimer block
