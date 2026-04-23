from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from src.engine.inference import BreastUltrasoundInferenceService

from app.components.result_panels import diagnosis_markdown
from app.components.status_panels import status_markdown, warnings_markdown


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
    service = BreastUltrasoundInferenceService.from_config(config_path)
    with gr.Blocks(title="BUCAD 乳腺超声智能辅助诊断系统") as demo:
        gr.Markdown(
            "\n".join(
                [
                    "# BUCAD 乳腺超声智能辅助诊断系统",
                    "上传一张乳腺超声图像，系统将输出良恶性概率、病灶定位叠加图和模型关注热力图。",
                    "> 本系统仅用于辅助分析，不能替代医生诊断。",
                ]
            )
        )
        with gr.Row():
            image_input = gr.Image(type="numpy", label="上传乳腺超声图像")
            with gr.Column():
                threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.01, label="恶性判定阈值")
                need_segmentation = gr.Checkbox(value=True, label="生成病灶定位图")
                need_explanation = gr.Checkbox(value=True, label="生成解释热力图")
                analyze_button = gr.Button("开始分析")

        status_output = gr.Markdown()
        diagnosis_output = gr.Markdown()
        with gr.Row():
            original_output = gr.Image(label="原始图像", type="numpy")
            lesion_output = gr.Image(label="病灶定位叠加图", type="numpy")
            explanation_output = gr.Image(label="模型解释热力图", type="numpy")
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
