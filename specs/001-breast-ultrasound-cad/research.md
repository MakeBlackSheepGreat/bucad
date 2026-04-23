# Phase 0 Research: 乳腺超声辅助诊断系统

## Decision 1: Use a single shared PyTorch codebase for training and inference

- **Decision**: Implement classification, segmentation, inference, and explanation in
  one Python codebase centered on PyTorch and shared preprocessing utilities.
- **Rationale**: The competition scope is narrow, the team is small, and the
  constitution favors direct progress toward a runnable deliverable. A shared pipeline
  minimizes duplicated preprocessing logic and keeps checkpoints, metric reports, and
  UI inference behavior consistent.
- **Alternatives considered**:
  - Separate training and inference repositories: rejected because it would duplicate
    configs, loaders, and artifact contracts.
  - Notebook-only workflow: rejected because it is weak for packaging, repeatability,
    and handoff.

## Decision 2: Use `timm.create_model(..., pretrained=True)` for the classifier

- **Decision**: Build the classification branch with timm image-classification
  backbones and use pretrained weights as the default baseline.
- **Rationale**: timm exposes a stable model factory API and pretrained backbone
  selection without requiring custom architecture code, which matches the constitution
  goal of using mature components over invention. It also keeps backbone experiments
  within a controlled interface.
- **Alternatives considered**:
  - Handwritten CNN baseline as the primary model: rejected because the spec targets
    diagnostic quality and explainable demos, not from-scratch architecture research.
  - Torchvision-only model zoo: rejected because timm offers broader backbone coverage
    with one consistent creation interface.

## Decision 3: Use `segmentation_models_pytorch.Unet` for lesion masks

- **Decision**: Implement lesion segmentation with
  `segmentation_models_pytorch.Unet`, using an ImageNet-pretrained encoder and a
  single-channel output mask.
- **Rationale**: SMP provides a compact segmentation abstraction while still exposing
  encoder choice, depth, and decoder settings. U-Net remains a strong fit for
  medical-style pixel localization with limited dataset scale.
- **Alternatives considered**:
  - Building a custom U-Net implementation: rejected because it adds maintenance cost
    without improving the deliverable.
  - Treating localization as bounding-box regression only: rejected because mask
    overlays are more useful for demonstration and clinical explanation.

## Decision 4: Use OpenCV + torchvision transforms for deterministic image pipelines

- **Decision**: Read and normalize ultrasound images with OpenCV, then convert to
  tensors and apply training/inference transforms through torchvision-compatible
  utilities.
- **Rationale**: OpenCV is already part of the user-selected stack and is well-suited
  to grayscale image loading, resizing, overlay rendering, and export. Torchvision
  transform guidance supports tensor- and image-based augmentation while preserving a
  clean path to deterministic seeded training.
- **Alternatives considered**:
  - Adding another augmentation framework as a hard dependency: rejected because the
    user explicitly fixed the stack for this planning phase.
  - PIL-only image handling: rejected because OpenCV better matches overlay and export
    needs for diagnostic visualization.

## Decision 5: Use AMP on CUDA, but keep CPU-safe fallback paths

- **Decision**: Use `torch.autocast` and `torch.amp.GradScaler("cuda")` during CUDA
  training, while keeping inference and smoke tests runnable on CPU.
- **Rationale**: PyTorch AMP is the native path for mixed-precision speedups and keeps
  model code close to standard training loops. The constitution also requires CPU
  fallback for onboarding and demo recovery.
- **Alternatives considered**:
  - Full fp32-only training: rejected because it wastes available GPU throughput.
  - GPU-only inference pipeline: rejected because it would break the demo-readiness
    gate when CUDA is unavailable.

## Decision 6: Standardize metric reporting in `artifacts/reports/`

- **Decision**: Store evaluation outputs as structured files under
  `artifacts/reports/`, including AUC, Sensitivity, Specificity, confusion values, and
  representative visualization outputs.
- **Rationale**: This directly satisfies the constitution’s metric gate and makes it
  possible to compare folds, thresholds, and demo artifacts without relying on console
  logs alone.
- **Alternatives considered**:
  - Terminal-only metric output: rejected because it is not durable or reviewable.
  - Spreadsheet-only reporting outside the repo flow: rejected because it weakens
    traceability between code, configs, and results.

## Decision 7: Use Grad-CAM as the default explanation layer for classifier outputs

- **Decision**: Generate explanation maps with `pytorch-grad-cam`, targeting the final
  spatial feature block of the chosen classifier backbone.
- **Rationale**: The library explicitly supports CNN classification workflows, batched
  usage, and smoothing options, which are sufficient for explanation overlays in a
  competition demo. It also keeps explanation logic decoupled from the main model.
- **Alternatives considered**:
  - Custom manual saliency implementation: rejected because it adds debugging burden.
  - Explanation-free MVP: rejected because the spec requires interpretable outputs.

## Decision 8: Deliver the UI with Gradio Blocks and `Image(type="numpy")`

- **Decision**: Implement the diagnostic prototype with Gradio Blocks, using image
  upload components that pass `numpy.ndarray` values into the inference function and
  image/markdown outputs for overlays and narrative explanation.
- **Rationale**: Gradio’s documented image component behavior maps directly onto the
  planned inference function signature, and Blocks gives enough layout control for a
  polished competition demo without introducing a separate frontend stack.
- **Alternatives considered**:
  - Desktop-native GUI toolkit: rejected because it increases UI complexity and slows
    iteration.
  - Separate REST backend and frontend: rejected because it adds deployment overhead
    without improving the core deliverable.

## Decision 9: Package Windows demos with PyInstaller in `onedir` mode first

- **Decision**: Treat `onedir` packaging as the primary demo target, with weights and
  configs copied or referenced beside the packaged app.
- **Rationale**: PyInstaller documentation highlights packaging pitfalls around bundled
  resources. `onedir` is easier to inspect, debug, and repair than `onefile`, which is
  important for a presentation-oriented prototype carrying local model assets.
- **Alternatives considered**:
  - `onefile` as the default package: rejected because runtime extraction and hidden
    import debugging are less transparent.
  - No packaging step: rejected because the spec explicitly requires a deliverable
    software prototype.

## Decision 10: Keep configuration in YAML and execution in explicit scripts

- **Decision**: Store path/model/runtime options in YAML files and expose each
  user-facing action through a small `argparse` script.
- **Rationale**: This keeps the novice-friendly command surface aligned with the
  handbook while still making runs reproducible and shareable across teammates.
- **Alternatives considered**:
  - Hardcoded paths in code: rejected because they fail the reproducibility gate.
  - Environment-variable-only configuration: rejected because it is harder to review
    and share in a small team.

## Sources

- PyTorch AMP: https://docs.pytorch.org/docs/stable/amp.html
- Torchvision transforms: https://docs.pytorch.org/vision/master/transforms.html
- timm model factory: https://huggingface.co/docs/timm/reference/models
- segmentation_models_pytorch U-Net: https://smp.readthedocs.io/en/v0.2.1/models.html
- Gradio Interface and Image docs: https://www.gradio.app/docs/gradio/interface
- Gradio Image component behavior: https://www.gradio.app/docs/gradio/image
- pytorch-grad-cam usage: https://github.com/jacobgil/pytorch-grad-cam
- PyInstaller pitfalls: https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html
