from __future__ import annotations

from html import escape

from app.components.localization import (
    localize_confidence_band,
    localize_disclaimer,
    localize_final_label,
    localize_recommendation,
)
from src.utils.results import InferenceResponse


def _risk_level(final_label: str, confidence_band: str) -> str:
    if final_label == "malignant" and confidence_band == "high":
        return "高风险"
    if confidence_band == "borderline":
        return "需复核"
    if final_label == "malignant":
        return "中高风险"
    return "低风险"


def _display_model_version(model_version: str, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    display_name = metadata.get("ensemble_display_name")
    if display_name:
        return str(display_name)
    if (
        model_version.startswith("mixed-ensemble:")
        and "convnext_tiny" in model_version
        and "tf_efficientnetv2_s" in model_version
    ):
        return "ConvNeXt-Tiny 主模型 + EfficientNetV2-S 双模型集成"
    if model_version.startswith("mixed-ensemble:") and "densenet121" in model_version:
        return "EfficientNetV2-S + DenseNet121 混合集成"
    if model_version.startswith("ensemble:") and "efficientnetv2_s" in model_version:
        return "EfficientNetV2-S 五折集成"
    return model_version


def diagnosis_markdown(response: InferenceResponse) -> str:
    if response.result is None:
        return """
<div class="result-card">
  <div class="section-title">诊断结果</div>
  <div class="empty-state">上传图像并点击“开始诊断”后显示结果。</div>
</div>
"""

    result = response.result
    malignant_pct = result.malignant_probability * 100
    benign_pct = result.benign_probability * 100
    final_label = localize_final_label(result.final_label)
    confidence = localize_confidence_band(result.confidence_band)
    risk_level = _risk_level(result.final_label, result.confidence_band)
    risk_class = "danger" if result.final_label == "malignant" else "safe"
    recommendation = escape(localize_recommendation(result.final_label, result.confidence_band))
    disclaimer = escape(localize_disclaimer(result.auxiliary_use_disclaimer))
    model_version = escape(_display_model_version(result.model_version, response.metadata))
    decision_threshold = float(response.metadata.get("decision_threshold", 0.5))

    return f"""
<div class="result-card">
  <div class="section-title">诊断结果</div>
  <div class="metric-grid">
    <div class="metric-card danger-soft">
      <div class="metric-label">恶性概率</div>
      <div class="metric-value danger-text">{malignant_pct:.1f}%</div>
      <div class="ring danger-ring" style="--value:{malignant_pct:.1f}"></div>
    </div>
    <div class="metric-card safe-soft">
      <div class="metric-label">良性概率</div>
      <div class="metric-value safe-text">{benign_pct:.1f}%</div>
      <div class="ring safe-ring" style="--value:{benign_pct:.1f}"></div>
    </div>
    <div class="metric-card verdict-card {risk_class}">
      <div class="metric-label">最终判定</div>
      <div class="verdict-text">{final_label}</div>
      <div class="risk-pill">{risk_level}</div>
    </div>
  </div>
  <div class="result-strip compact">
    <span>阈值 <b>{decision_threshold:.3f}</b></span>
    <span>置信度 <b>{confidence}</b></span>
    <span class="model-chip" title="{model_version}">模型 <b>{model_version}</b></span>
  </div>
  <div class="recommendation-box">{recommendation}</div>
  <div class="disclaimer-box">{disclaimer}</div>
</div>
"""
