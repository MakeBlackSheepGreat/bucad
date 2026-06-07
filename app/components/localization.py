"""Chinese UI labels and clinical wording helpers for diagnosis results."""

from __future__ import annotations

from src.utils.results import AUXILIARY_USE_DISCLAIMER


STATUS_TEXT = {
    "completed": "推理完成",
    "quality_blocked": "图像质量不足",
    "invalid_input": "输入无效",
    "classification_unavailable": "分类模型不可用",
    "unexpected_runtime_error": "运行异常",
}

FINAL_LABEL_TEXT = {
    "benign": "良性",
    "malignant": "恶性",
}

CONFIDENCE_BAND_TEXT = {
    "high": "高",
    "low": "中低",
    "borderline": "边界",
}

WARNING_TEXT = {
    "Image quality is too poor for reliable analysis.": "图像质量过低，当前无法给出可靠分析结果。",
    "No classifier checkpoint is configured.": "未配置分类模型权重，请检查推理配置。",
    "Torch is not available in the current environment.": "当前环境未安装 PyTorch，无法加载模型。",
    "Classifier input tensor conversion failed.": "分类模型输入张量转换失败。",
    "No segmentation checkpoint is configured.": "未配置分割模型权重，病灶定位图无法生成。",
    "Segmentation is disabled in runtime config.": "当前配置已关闭病灶定位输出。",
    "Torch is not available for segmentation.": "当前环境未安装 PyTorch，无法运行分割模型。",
    "Segmentation input tensor conversion failed.": "分割模型输入张量转换失败。",
    "Grad-CAM is disabled in runtime config.": "当前配置已关闭 Grad-CAM 热力图。",
    "Classifier model is not loaded for explanation.": "分类模型尚未加载，无法生成解释热力图。",
    "Torch tensor conversion failed for explanation.": "解释模块输入张量转换失败。",
    "Grad-CAM dependency is not installed in the current environment.": "当前环境未安装 Grad-CAM 依赖，无法生成热力图。",
    "Could not resolve a Grad-CAM target layer.": "无法定位 Grad-CAM 目标层。",
    "Grad-CAM returned no explanation map.": "Grad-CAM 未返回有效热力图。",
    "No image backend is installed. Install opencv-python or Pillow.": "当前缺少图像读取依赖，请安装 opencv-python 或 Pillow。",
    "No image backend is installed for saving.": "当前缺少图像保存依赖，无法导出可视化结果。",
    AUXILIARY_USE_DISCLAIMER: "本系统仅用于辅助分析和原型演示，不能替代医生诊断。",
}


def localize_status(status: str) -> str:
    """Translate an inference status code into Chinese UI text."""
    return STATUS_TEXT.get(status, status)


def localize_final_label(label: str) -> str:
    """Translate a final diagnosis label into Chinese UI text."""
    return FINAL_LABEL_TEXT.get(label, label)


def localize_confidence_band(band: str) -> str:
    """Translate a confidence band into Chinese UI text."""
    return CONFIDENCE_BAND_TEXT.get(band, band)


def localize_recommendation(final_label: str, confidence_band: str) -> str:
    """Return the Chinese recommendation text for a label/confidence pair."""
    localized_label = localize_final_label(final_label)
    if confidence_band == "borderline":
        return "结果接近判定阈值，建议人工复核并结合原始超声图像判断。"
    if confidence_band == "low":
        return f"模型当前倾向于{localized_label}，但置信度有限，建议谨慎复核。"
    if final_label == "malignant":
        return "系统提示高风险恶性征象，建议尽快进行人工复核。"
    return "系统当前更倾向于良性，但仍需结合医生判断。"


def localize_disclaimer(message: str) -> str:
    """Translate known disclaimer text into Chinese."""
    return WARNING_TEXT.get(message, message)


def localize_warning(message: str) -> str:
    """Translate known warning messages while preserving dynamic suffix text."""
    if message in WARNING_TEXT:
        return WARNING_TEXT[message]
    prefix_map = {
        "Image does not exist: ": "图像文件不存在：",
        "Unsupported image file type: ": "暂不支持该图像格式：",
        "Unable to read image: ": "无法读取图像文件：",
        "Unable to encode image for saving: ": "无法编码并保存图像：",
        "Unexpected image shape: ": "图像维度不符合预期：",
        "Unexpected runtime error: ": "运行时出现异常：",
    }
    for prefix, localized in prefix_map.items():
        if message.startswith(prefix):
            return f"{localized}{message.removeprefix(prefix)}"
    return message
