from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from app.components.result_panels import diagnosis_markdown
from app.components.status_panels import status_markdown, warnings_markdown
from src.engine.inference import BreastUltrasoundInferenceService


DEFAULT_THRESHOLD = 0.51


def bundled_resource_path(relative_path: str | Path) -> Path:
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    bundled_path = base_dir / relative_path
    if bundled_path.exists():
        return bundled_path
    return Path(relative_path)


def build_theme() -> gr.themes.ThemeClass:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        color_accent="#2563eb",
        color_accent_soft="#dbeafe",
        border_color_accent="#93b8ff",
        border_color_accent_subdued="#dbe7ff",
        loader_color="#2563eb",
        slider_color="#2563eb",
        checkbox_background_color_selected="#2563eb",
        checkbox_border_color_focus="#2563eb",
        checkbox_border_color_selected="#2563eb",
        checkbox_label_background_fill_selected="#eff6ff",
        checkbox_label_border_color_selected="#93b8ff",
        checkbox_label_text_color_selected="#172554",
        button_primary_background_fill="linear-gradient(135deg, #2563eb, #1d4ed8)",
        button_primary_background_fill_hover="linear-gradient(135deg, #1d4ed8, #1e40af)",
    )


APP_CSS = """
:root {
  --primary: #2563eb;
  --primary-dark: #1e3a8a;
  --primary-soft: #60a5fa;
  --blue-50: #eff6ff;
  --blue-100: #dbeafe;
  --border: #dbe7ff;
  --text: #172554;
  --muted: #64748b;
  --danger: #e11d48;
  --danger-bg: #fff1f2;
  --safe: #0f766e;
  --safe-bg: #ecfeff;
  --panel: rgba(255, 255, 255, 0.92);
}

.gradio-container {
  --color-accent: var(--primary) !important;
  --color-accent-soft: var(--blue-100) !important;
  --border-color-accent: #93b8ff !important;
  --border-color-accent-subdued: var(--border) !important;
  --color-accent-soft-hover: #bfdbfe !important;
  --color-accent-soft-active: #93c5fd !important;
  --button-primary-background-fill: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
  --button-primary-background-fill-hover: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
  --loader-color: var(--primary) !important;
  --checkbox-background-color-selected: var(--primary) !important;
  --checkbox-border-color-focus: var(--primary) !important;
  --checkbox-border-color-selected: var(--primary) !important;
  --checkbox-label-background-fill-selected: var(--blue-50) !important;
  --checkbox-label-border-color-selected: #93b8ff !important;
  --checkbox-label-text-color-selected: var(--text) !important;
  --slider-color: var(--primary) !important;
  accent-color: var(--primary);
  max-width: 1600px !important;
  min-height: 100vh;
  background:
    radial-gradient(circle at 12% 8%, rgba(37, 99, 235, 0.10), transparent 26%),
    linear-gradient(135deg, #f8fbff 0%, #eef5ff 50%, #f8fbff 100%);
  color: var(--text);
  font-family: "Microsoft YaHei", "Inter", "Segoe UI", sans-serif;
}

.app-shell {
  min-height: 100vh;
  padding: clamp(6px, 1vw, 12px);
  display: flex;
  flex-direction: column;
  gap: clamp(8px, 1vw, 12px);
}

.hero {
  display: flex;
  gap: 12px;
  align-items: center;
  flex: 0 0 auto;
}

.logo-mark {
  width: clamp(40px, 3.2vw, 48px);
  height: clamp(40px, 3.2vw, 48px);
  display: grid;
  place-items: center;
  color: var(--primary);
  border-radius: 15px;
  background: linear-gradient(145deg, #eaf2ff, #ffffff);
  border: 1px solid var(--border);
  font-size: clamp(22px, 2vw, 26px);
  box-shadow: 0 10px 26px rgba(37, 99, 235, 0.12);
}

.hero h1 {
  margin: 0;
  color: #102a6b;
  font-size: clamp(21px, 1.9vw, 27px);
  letter-spacing: 0.5px;
}

.hero p {
  margin: 2px 0 0;
  color: #365486;
  font-size: clamp(12px, 0.95vw, 14px);
}

.workspace-row {
  flex: 1 1 auto;
  min-height: 0;
  display: grid !important;
  grid-template-columns: minmax(330px, 0.92fr) minmax(620px, 1.9fr);
  gap: clamp(10px, 1vw, 14px) !important;
  align-items: stretch;
}

.left-panel,
.right-panel,
.result-card,
.status-card,
.viewer-card,
.mini-card {
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--panel);
  box-shadow: 0 18px 48px rgba(30, 64, 175, 0.08);
}

.left-panel {
  min-width: 0 !important;
  min-height: 0;
  padding: clamp(10px, 1vw, 14px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.right-panel {
  min-width: 0 !important;
  min-height: 0;
  padding: clamp(10px, 1vw, 14px);
  display: grid !important;
  grid-template-rows: auto minmax(0, 1fr) auto;
  overflow: hidden;
}

.section-title {
  color: #0f3a8a;
  font-weight: 800;
  font-size: clamp(15px, 1vw, 17px);
  margin-bottom: 7px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-title::before {
  content: "";
  width: 8px;
  height: 16px;
  border-radius: 6px;
  background: linear-gradient(180deg, #60a5fa, #2563eb);
}

.upload-box {
  border: 1.5px dashed #93b8ff !important;
  border-radius: 18px !important;
  background: rgba(239, 246, 255, 0.75) !important;
}

.upload-box button,
.upload-box svg,
.upload-box [role="button"],
.upload-box .icon,
.upload-box [aria-label*="upload" i],
.upload-box [aria-label*="上传" i] {
  color: var(--primary) !important;
  stroke: var(--primary) !important;
}

.upload-box button:hover,
.upload-box [role="button"]:hover {
  color: var(--primary-dark) !important;
  background: #eff6ff !important;
}

.left-panel input[type="checkbox"],
.left-panel input[type="range"] {
  accent-color: var(--primary) !important;
}

.left-panel input[type="checkbox"]:checked {
  background-color: var(--primary) !important;
  border-color: var(--primary) !important;
}

.left-panel input[type="checkbox"]:checked + *,
.left-panel label:has(input[type="checkbox"]:checked) {
  color: #172554 !important;
}

.left-panel [role="checkbox"][aria-checked="true"] {
  background-color: var(--primary) !important;
  border-color: var(--primary) !important;
}

.left-panel input[type="range"]::-webkit-slider-runnable-track {
  accent-color: var(--primary) !important;
}

.left-panel input[type="range"]::-webkit-slider-thumb {
  background: #ffffff !important;
  border: 2px solid var(--primary) !important;
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
}

.left-panel input[type="range"]::-moz-range-progress {
  background: var(--primary) !important;
}

.left-panel input[type="range"]::-moz-range-thumb {
  background: #ffffff !important;
  border: 2px solid var(--primary) !important;
}

.control-caption {
  color: var(--muted);
  font-size: 12px;
  margin: -4px 0 6px;
}

.primary-btn {
  background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
  color: white !important;
  border-radius: 12px !important;
  border: none !important;
  box-shadow: 0 12px 26px rgba(37, 99, 235, 0.28) !important;
}

.secondary-btn {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
}

.result-card,
.status-card {
  padding: clamp(9px, 0.8vw, 12px);
  margin-top: 8px;
}

.metric-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 7px;
}

.metric-card {
  min-height: clamp(82px, 10vh, 102px);
  border-radius: 13px;
  padding: clamp(8px, 0.75vw, 10px);
  border: 1px solid var(--border);
  text-align: center;
  position: relative;
  overflow: hidden;
}

.danger-soft { background: var(--danger-bg); }
.safe-soft { background: var(--safe-bg); }
.danger-text { color: var(--danger); }
.safe-text { color: #14b8a6; }

.metric-label {
  color: #315179;
  font-weight: 700;
  font-size: 12px;
}

.metric-value {
  font-size: clamp(21px, 1.8vw, 26px);
  font-weight: 900;
  margin-top: 8px;
}

.ring {
  height: 6px;
  border-radius: 99px;
  margin-top: 8px;
  background: linear-gradient(90deg, currentColor calc(var(--value) * 1%), rgba(148, 163, 184, 0.18) 0);
}

.danger-ring { color: var(--danger); }
.safe-ring { color: #14b8a6; }

.verdict-card.danger {
  background: linear-gradient(160deg, #fff1f2, #ffffff);
  border-color: #fecdd3;
}

.verdict-card.safe {
  background: linear-gradient(160deg, #ecfeff, #ffffff);
  border-color: #a5f3fc;
}

.verdict-text {
  font-size: clamp(25px, 2.3vw, 32px);
  font-weight: 950;
  margin-top: 6px;
}

.verdict-card.danger .verdict-text { color: var(--danger); }
.verdict-card.safe .verdict-text { color: var(--safe); }

.risk-pill {
  display: inline-block;
  margin-top: 4px;
  padding: 3px 9px;
  border-radius: 99px;
  background: rgba(255,255,255,0.8);
  color: #1e3a8a;
  font-weight: 700;
}

.result-strip {
  margin-top: 7px;
  display: flex;
  justify-content: flex-start;
  gap: 6px;
  flex-wrap: wrap;
  background: #f8fbff;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 6px;
  color: #315179;
}

.result-strip span {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
  padding: 5px 8px;
  border-radius: 10px;
  background: #ffffff;
  border: 1px solid #e5eefc;
  font-size: 12px;
}

.result-strip .model-chip {
  max-width: 100%;
  flex: 1 1 100%;
  align-items: flex-start;
  white-space: normal;
  overflow: visible;
  overflow-wrap: anywhere;
  line-height: 1.4;
}

.result-strip .model-chip b {
  white-space: normal;
  overflow-wrap: anywhere;
}

.recommendation-box,
.disclaimer-box,
.hint-text {
  margin-top: 7px;
  color: #315179;
  line-height: 1.45;
  font-size: 13px;
}

.disclaimer-box {
  padding: 7px 9px;
  border-radius: 12px;
  background: #f8fafc;
  color: #64748b;
}

.status-line {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 99px;
  font-weight: 800;
}

.status-line.ok {
  color: #047857;
  background: #ecfdf5;
}

.status-line.warn {
  color: #b45309;
  background: #fffbeb;
}

.status-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: currentColor;
}

.warning-list {
  margin: 0;
  padding-left: 20px;
  color: #9f1239;
  line-height: 1.45;
  font-size: 13px;
}

.viewer-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #0f3a8a;
  font-weight: 900;
  margin-bottom: 8px;
  min-height: 24px;
}

.image-frame img {
  border-radius: 14px !important;
  object-fit: contain !important;
  background: #020617 !important;
}

.mini-card {
  min-width: 0 !important;
  padding: 7px;
}

.footer-note {
  color: #64748b;
  text-align: center;
  font-size: 12px;
  margin-top: 6px;
}

.main-image-card {
  min-height: 0;
  padding: 7px;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: #fff;
}

.image-grid {
  min-height: 0;
  display: grid !important;
  grid-template-rows: minmax(240px, 1.7fr) minmax(150px, 0.85fr);
  gap: 8px !important;
}

.mini-image-row {
  min-height: 0;
  gap: 8px !important;
}

.compact-image,
.compact-image > div,
.compact-image .wrap,
.compact-image .image-container {
  height: 100% !important;
  min-height: 0 !important;
}

.compact-image .image-container {
  display: flex !important;
  align-items: center;
  justify-content: center;
}

.compact-image img {
  max-height: 100% !important;
}

.left-panel .form,
.left-panel .block,
.right-panel .block {
  min-width: 0 !important;
}

.left-panel .gap,
.right-panel .gap {
  gap: 8px !important;
}

@media (max-height: 850px) and (min-width: 1101px) {
  .app-shell { gap: 6px; padding: 6px 10px; }
  .logo-mark { width: 38px; height: 38px; font-size: 21px; }
  .hero h1 { font-size: 21px; }
  .hero p { font-size: 12px; }
  .left-panel, .right-panel { padding: 10px; border-radius: 16px; }
  .section-title { font-size: 15px; margin-bottom: 5px; }
  .result-card, .status-card { margin-top: 6px; padding: 8px; border-radius: 14px; }
  .metric-card { min-height: 76px; padding: 7px; }
  .metric-value { margin-top: 5px; font-size: 21px; }
  .verdict-text { margin-top: 4px; font-size: 25px; }
  .ring { margin-top: 6px; }
  .recommendation-box, .disclaimer-box, .hint-text { font-size: 12px; line-height: 1.35; }
  .result-strip { margin-top: 6px; padding: 5px; }
  .result-strip span { padding: 4px 7px; font-size: 12px; }
  .image-grid { grid-template-rows: minmax(220px, 1.75fr) minmax(130px, 0.78fr); gap: 6px !important; }
  .mini-image-row { gap: 6px !important; }
  .main-image-card, .mini-card { padding: 6px; border-radius: 14px; }
  .footer-note { margin-top: 4px; font-size: 11px; }
}

@media (max-width: 1100px) {
  .app-shell { min-height: auto; }
  .workspace-row {
    display: flex !important;
    flex-direction: column;
    grid-template-columns: 1fr;
  }
  .left-panel,
  .right-panel {
    overflow: visible;
  }
  .image-grid {
    grid-template-rows: auto auto;
  }
  .mini-image-row {
    flex-direction: column;
  }
}

@media (max-width: 640px) {
  .hero { align-items: flex-start; }
  .hero p { display: none; }
  .metric-grid { grid-template-columns: 1fr; }
  .result-strip .model-chip { flex-basis: auto; }
}
"""


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


def build_app(config_path: str | Path | None = None):
    if config_path is None:
        config_path = bundled_resource_path("configs/inference/demo.yml")
    service = BreastUltrasoundInferenceService.from_config(config_path)
    default_threshold = float(service.runtime_config.get("default_threshold", DEFAULT_THRESHOLD))
    ensemble_display_name = str(
        service.runtime_config.get(
            "ensemble_display_name",
            "ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate",
        )
    )
    with gr.Blocks(
        title="乳腺超声肿瘤良恶性分类辅助诊断系统（BUCAD）",
    ) as demo:
        gr.HTML(f"<style>{APP_CSS}</style>")
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                f"""
<div class="hero">
  <div class="logo-mark">⌁</div>
  <div>
    <h1>乳腺超声肿瘤良恶性分类辅助诊断系统（BUCAD）</h1>
    <p>Breast Ultrasound Computer-Aided Diagnosis · {ensemble_display_name} · ROI 定位 · Grad-CAM 解释</p>
  </div>
</div>
"""
            )

            with gr.Row(equal_height=True, elem_classes=["workspace-row"]):
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
                        clear_button = gr.ClearButton(
                            value="清空",
                            components=[image_input],
                            elem_classes=["secondary-btn"],
                            scale=1,
                        )

                    diagnosis_output = gr.HTML(
                        """
<div class="result-card">
  <div class="section-title">诊断结果</div>
  <div class="empty-state">上传图像并点击“开始诊断”后显示结果。</div>
</div>
"""
                    )
                    status_output = gr.HTML(
                        """
<div class="status-card">
  <div class="section-title">运行状态</div>
  <div class="hint-text">等待上传图像。</div>
</div>
"""
                    )
                    warning_output = gr.HTML(
                        """
<div class="status-card">
  <div class="section-title">辅助说明</div>
  <div class="hint-text">本系统仅用于辅助分析和原型演示，不能替代医生诊断。</div>
</div>
"""
                    )

                with gr.Column(scale=8, min_width=680, elem_classes=["right-panel"]):
                    gr.HTML(
                        """
<div class="viewer-title">
  <span>图像分析结果</span>
  <span>原图 / 分割 / 热力图</span>
</div>
"""
                    )
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
                    gr.HTML(
                        '<div class="footer-note">提示：分割图和热力图用于辅助理解模型输出，不代表临床标注或最终诊断。</div>'
                    )

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
    app.launch(inbrowser=True, show_error=True, theme=build_theme())


if __name__ == "__main__":
    main()
