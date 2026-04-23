# UI Review Contract: Diagnostic Demo Flow

## Purpose

Define the minimum behavior expected from the demonstration interface so planning and
implementation stay aligned on what a “complete demo” means.

## Primary Flow

1. User opens the application and sees a single-image diagnostic entry point.
2. User uploads one breast ultrasound image.
3. User triggers analysis.
4. System validates input and shows progress state.
5. System returns diagnosis, lesion visualization, and explanation outputs in one
   reviewable page.

## Required UI Regions

- **Input region**: image upload plus optional threshold or mode controls
- **Status region**: loading, success, partial-success, or failure state
- **Diagnosis region**: benign/malignant probabilities and final label
- **Lesion region**: mask, overlay, contour, or equivalent localization output
- **Explanation region**: Grad-CAM or equivalent model-attention visualization
- **Safety region**: disclaimer, warning, and manual-review recommendation when needed

## Interaction Rules

- The user must not need to navigate to another page to see the analysis result.
- The UI must clearly separate original image, localization view, and explanation view.
- Borderline or low-confidence outcomes must be visually distinguished from routine outcomes.
- Missing optional outputs must be explained in-place, not hidden.

## Review Expectations

- A reviewer should understand the result without reading terminal output.
- A non-developer should be able to complete one end-to-end analysis after following
  the quickstart steps.
- Screenshots captured from the UI should be sufficient for report and defense
  materials.
