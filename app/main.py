from __future__ import annotations

from pathlib import Path

from src.engine.inference import BreastUltrasoundInferenceService
from src.utils.runtime import optional_import

from app.components.result_panels import diagnosis_markdown
from app.components.status_panels import status_markdown, warnings_markdown


gr = optional_import("gradio")


def analyze_upload(
    image,
    threshold: float,
    need_segmentation: bool,
    need_explanation: bool,
    *,
    service: BreastUltrasoundInferenceService,
):
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


def build_app(config_path: str | Path = "configs/inference/demo.yml"):
    if gr is None:
        raise RuntimeError("Gradio is not installed. Install requirements before launching the UI.")

    service = BreastUltrasoundInferenceService.from_config(config_path)
    with gr.Blocks(title="BUCAD Demo") as demo:
        gr.Markdown("# BUCAD Breast Ultrasound CAD")
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Breast Ultrasound Image")
            with gr.Column():
                threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.01, label="Decision threshold")
                need_segmentation = gr.Checkbox(value=True, label="Need lesion localization")
                need_explanation = gr.Checkbox(value=True, label="Need explanation map")
                analyze_button = gr.Button("Analyze")

        status_output = gr.Markdown()
        diagnosis_output = gr.Markdown()
        with gr.Row():
            original_output = gr.Image(label="Original", type="numpy")
            lesion_output = gr.Image(label="Lesion Overlay", type="numpy")
            explanation_output = gr.Image(label="Explanation", type="numpy")
        warning_output = gr.Markdown()

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
    return demo


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
