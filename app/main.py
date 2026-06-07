"""Gradio application assembly for the BUCAD diagnostic demo."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from app.components.layout import (
    EMPTY_DIAGNOSIS_HTML,
    EMPTY_STATUS_HTML,
    EMPTY_WARNING_HTML,
    footer_note_html,
    hero_html,
    viewer_header_html,
)
from app.components.result_panels import diagnosis_markdown
from app.components.status_panels import status_markdown, warnings_markdown
from app.components.theme import APP_CSS, build_theme
from src.engine.inference import BreastUltrasoundInferenceService


DEFAULT_THRESHOLD = 0.51


@dataclass(slots=True)
class _ControlPanel:
    """Grouped references for the upload, action, and text-result controls."""

    image_input: gr.Image
    threshold: gr.Slider
    need_segmentation: gr.Checkbox
    need_explanation: gr.Checkbox
    analyze_button: gr.Button
    clear_button: gr.Button
    diagnosis_output: gr.HTML
    status_output: gr.HTML
    warning_output: gr.HTML


@dataclass(slots=True)
class _ViewerPanel:
    """Grouped references for image outputs rendered in the diagnostic workspace."""

    original_output: gr.Image
    lesion_output: gr.Image
    explanation_output: gr.Image


def bundled_resource_path(relative_path: str | Path) -> Path:
    """Resolve a resource path inside a PyInstaller bundle or the source tree."""
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    bundled_path = base_dir / relative_path
    if bundled_path.exists():
        return bundled_path
    return Path(relative_path)


def analyze_upload(
    image,
    threshold: float,
    need_segmentation: bool,
    need_explanation: bool,
    *,
    service: BreastUltrasoundInferenceService,
):
    """Run a Gradio upload through the inference service and return UI panel values."""
    response = service.diagnose(
        image,
        decision_threshold=threshold,
        need_segmentation=need_segmentation,
        need_explanation=need_explanation,
    )
    return (
        status_markdown(response),
        diagnosis_markdown(response),
        response.original_image_view,
        response.lesion_overlay_view,
        response.explanation_view,
        warnings_markdown(response),
    )


def reset_workspace():
    """Restore the demo to its initial empty component state."""
    return (
        None,
        EMPTY_STATUS_HTML,
        EMPTY_DIAGNOSIS_HTML,
        None,
        None,
        None,
        EMPTY_WARNING_HTML,
    )


def _runtime_display_name(service: BreastUltrasoundInferenceService) -> str:
    """Return the user-facing ensemble name from runtime config defaults."""
    return str(
        service.runtime_config.get(
            "ensemble_display_name",
            "ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate",
        )
    )


def _runtime_default_threshold(service: BreastUltrasoundInferenceService) -> float:
    """Return the configured malignancy threshold used to initialize the slider."""
    return float(service.runtime_config.get("default_threshold", DEFAULT_THRESHOLD))


def _bind_app_events(
    *,
    service: BreastUltrasoundInferenceService,
    analyze_button,
    clear_button,
    image_input,
    threshold,
    need_segmentation,
    need_explanation,
    status_output,
    diagnosis_output,
    original_output,
    lesion_output,
    explanation_output,
    warning_output,
) -> None:
    """Wire Gradio actions to inference and reset callbacks."""
    analyze_button.click(
        fn=lambda image, decision_threshold, segmentation, explanation: analyze_upload(
            image,
            decision_threshold,
            segmentation,
            explanation,
            service=service,
        ),
        inputs=[image_input, threshold, need_segmentation, need_explanation],
        outputs=[
            status_output,
            diagnosis_output,
            original_output,
            lesion_output,
            explanation_output,
            warning_output,
        ],
    )
    clear_button.click(
        fn=reset_workspace,
        inputs=None,
        outputs=[
            image_input,
            status_output,
            diagnosis_output,
            original_output,
            lesion_output,
            explanation_output,
            warning_output,
        ],
        queue=False,
    )


def _build_control_panel(*, default_threshold: float) -> _ControlPanel:
    """Create upload controls, toggles, action buttons, and text outputs."""
    with gr.Column(scale=4, min_width=390, elem_classes=["left-panel"]):
        gr.HTML('<div class="section-title">图像输入与诊断设置</div>')
        image_input = gr.Image(
            type="numpy",
            label="点击或拖拽上传乳腺超声图像",
            elem_classes=["upload-box"],
            height=180,
        )
        gr.HTML('<div class="control-caption">支持常见 JPG / PNG 图像。DICOM 需先转换为普通图像后上传。</div>')

        threshold = gr.Slider(
            0.1,
            0.9,
            value=default_threshold,
            step=0.001,
            label="恶性判定阈值",
            info=f"当前主线推荐 {default_threshold:.3f}；分割器为 UNet-ResNet18，ROI 使用 0.40 mask 阈值、最大连通域与面积质量门控。",
        )
        with gr.Row():
            need_segmentation = gr.Checkbox(value=True, label="生成病灶定位图")
            need_explanation = gr.Checkbox(value=True, label="生成 Grad-CAM 热力图")

        with gr.Row():
            analyze_button = gr.Button(
                "▶ 开始诊断",
                variant="primary",
                elem_classes=["primary-btn"],
                scale=2,
            )
            clear_button = gr.Button(
                value="清空",
                elem_classes=["secondary-btn"],
                scale=1,
            )

        diagnosis_output = gr.HTML(EMPTY_DIAGNOSIS_HTML)
        status_output = gr.HTML(EMPTY_STATUS_HTML)
        warning_output = gr.HTML(EMPTY_WARNING_HTML)

    return _ControlPanel(
        image_input=image_input,
        threshold=threshold,
        need_segmentation=need_segmentation,
        need_explanation=need_explanation,
        analyze_button=analyze_button,
        clear_button=clear_button,
        diagnosis_output=diagnosis_output,
        status_output=status_output,
        warning_output=warning_output,
    )


def _build_viewer_panel() -> _ViewerPanel:
    """Create image viewers for original, lesion overlay, and Grad-CAM output."""
    with gr.Column(scale=8, min_width=680, elem_classes=["right-panel"]):
        gr.HTML(viewer_header_html())
        with gr.Column(elem_classes=["image-grid"]):
            with gr.Column(elem_classes=["main-image-card"]):
                original_output = gr.Image(
                    label="原始超声图像",
                    type="numpy",
                    height=None,
                    elem_classes=["image-frame", "compact-image"],
                )
            with gr.Row(equal_height=True, elem_classes=["mini-image-row"]):
                with gr.Column(elem_classes=["mini-card"]):
                    lesion_output = gr.Image(
                        label="病灶定位叠加图",
                        type="numpy",
                        height=None,
                        elem_classes=["image-frame", "compact-image"],
                    )
                with gr.Column(elem_classes=["mini-card"]):
                    explanation_output = gr.Image(
                        label="模型关注区域（Grad-CAM）",
                        type="numpy",
                        height=None,
                        elem_classes=["image-frame", "compact-image"],
                    )
        gr.HTML(footer_note_html())

    return _ViewerPanel(
        original_output=original_output,
        lesion_output=lesion_output,
        explanation_output=explanation_output,
    )


def build_app(config_path: str | Path | None = None):
    """Build the BUCAD Gradio Blocks app without launching a server."""
    if config_path is None:
        config_path = bundled_resource_path("configs/inference/demo.yml")
    service = BreastUltrasoundInferenceService.from_config(config_path)
    default_threshold = _runtime_default_threshold(service)
    ensemble_display_name = _runtime_display_name(service)

    with gr.Blocks(
        title="乳腺超声肿瘤良恶性分类辅助诊断系统（BUCAD）",
        fill_width=True,
    ) as demo:
        gr.HTML(f"<style>{APP_CSS}</style>")
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(hero_html(ensemble_display_name))

            with gr.Row(equal_height=True, elem_classes=["workspace-row"]):
                controls = _build_control_panel(default_threshold=default_threshold)
                viewer = _build_viewer_panel()

            _bind_app_events(
                service=service,
                analyze_button=controls.analyze_button,
                clear_button=controls.clear_button,
                image_input=controls.image_input,
                threshold=controls.threshold,
                need_segmentation=controls.need_segmentation,
                need_explanation=controls.need_explanation,
                status_output=controls.status_output,
                diagnosis_output=controls.diagnosis_output,
                original_output=viewer.original_output,
                lesion_output=viewer.lesion_output,
                explanation_output=viewer.explanation_output,
                warning_output=controls.warning_output,
            )
    return demo


def main() -> None:
    """Launch the Gradio demo with the packaged/default inference config."""
    app = build_app()
    app.launch(inbrowser=True, show_error=True, theme=build_theme())


if __name__ == "__main__":
    main()
