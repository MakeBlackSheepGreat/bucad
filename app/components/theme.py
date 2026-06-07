from __future__ import annotations

import gradio as gr


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
  --checkbox-background-color_selected: var(--primary) !important;
  --checkbox-background-color-selected: var(--primary) !important;
  --checkbox-border-color-focus: var(--primary) !important;
  --checkbox-border-color-selected: var(--primary) !important;
  --checkbox-label-background-fill-selected: var(--blue-50) !important;
  --checkbox-label-border-color-selected: #93b8ff !important;
  --checkbox-label-text-color-selected: var(--text) !important;
  --slider-color: var(--primary) !important;
  accent-color: var(--primary);
  width: 100% !important;
  max-width: none !important;
  min-width: 0 !important;
  min-height: 100dvh;
  margin: 0 !important;
  padding: 0 !important;
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
  width: 100%;
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
  overflow: hidden;
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
  overflow: hidden;
}

.image-grid {
  min-height: 0;
  display: grid !important;
  grid-template-rows: minmax(clamp(260px, 48dvh, 620px), 1.8fr) minmax(clamp(150px, 24dvh, 300px), 0.82fr);
  gap: clamp(8px, 0.8vw, 12px) !important;
  overflow: hidden;
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
  overflow: hidden !important;
}

.compact-image .image-container {
  display: flex !important;
  align-items: center;
  justify-content: center;
}

.compact-image img {
  width: 100% !important;
  height: 100% !important;
  max-height: 100% !important;
  max-width: 100% !important;
  object-fit: contain !important;
  display: block !important;
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
