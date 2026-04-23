from __future__ import annotations

from src.utils.results import AUXILIARY_USE_DISCLAIMER


STATUS_TEXT = {
    "completed": "分析完成",
    "partial": "分析完成（部分可视化未生成）",
    "invalid_input": "输入无效",
    "quality_blocked": "图像质量不足",
    "classification_unavailable": "诊断不可用",
    "failed": "分析失败",
    "unexpected_runtime_error": "运行异常",
}

FINAL_LABEL_TEXT = {
    "benign": "良性",
    "malignant": "恶性",
}

CONFIDENCE_BAND_TEXT = {
    "high": "高",
    "low": "低",
    "borderline": "临界",
}

WARNING_TEXT = {
    "Uploaded image is invalid or too small.": "上传图像无效，或图像尺寸过小。",
    "Image quality is too poor for reliable analysis.": "图像质量过低，当前无法给出可靠分析结果。",
    "No classifier checkpoint is configured.": "未配置分类模型权重，无法执行诊断。",
    "Torch is not available in the current environment.": "当前环境缺少 Torch，无法执行诊断。",
    "Torch tensor conversion failed for classifier input.": "分类输入张量转换失败，无法执行诊断。",
    "Segmentation weights are not available.": "当前未提供病灶分割权重。",
    "Torch is not available for segmentation.": "当前环境缺少 Torch，无法生成病灶定位结果。",
    "Torch tensor conversion failed for segmentation.": "分割输入张量转换失败，无法生成病灶定位结果。",
    "Grad-CAM is disabled in runtime config.": "当前配置已禁用解释热力图。",
    "Classifier model is not loaded for explanation.": "分类模型未加载，无法生成解释热力图。",
    "Torch tensor conversion failed for explanation.": "解释分支输入张量转换失败，无法生成解释热力图。",
    "Grad-CAM dependency is not installed in the current environment.": "当前环境未安装 Grad-CAM 依赖，无法生成解释热力图。",
    "Could not resolve a Grad-CAM target layer.": "无法定位 Grad-CAM 目标层，无法生成解释热力图。",
    "Grad-CAM returned no explanation map.": "Grad-CAM 未返回有效热力图。",
    "No image backend is installed. Install opencv-python or Pillow.": "当前缺少图像读取依赖，请安装 opencv-python 或 Pillow。",
    "No image backend is installed for saving.": "当前缺少图像保存依赖，无法导出可视化结果。",
    AUXILIARY_USE_DISCLAIMER: "本系统仅用于辅助分析，不能替代医生诊断。",
}


def localize_status(status: str) -> str:
    return STATUS_TEXT.get(status, status)


def localize_final_label(label: str) -> str:
    return FINAL_LABEL_TEXT.get(label, label)


def localize_confidence_band(band: str) -> str:
    return CONFIDENCE_BAND_TEXT.get(band, band)


def localize_recommendation(final_label: str, confidence_band: str) -> str:
    localized_label = localize_final_label(final_label)
    if confidence_band == "borderline":
        return "结果接近判定阈值，建议由医生进一步复核。"
    if confidence_band == "low":
        return f"模型当前更倾向于{localized_label}，但置信度有限，建议结合原图谨慎判断。"
    if final_label == "malignant":
        return "系统提示高风险恶性征象，建议尽快由医生复核。"
    return "系统当前更倾向于良性，但仍需结合医生判断。"


def localize_disclaimer(message: str) -> str:
    return WARNING_TEXT.get(message, message)


def localize_warning(message: str) -> str:
    if message in WARNING_TEXT:
        return WARNING_TEXT[message]
    if message.startswith("Image does not exist: "):
        return f"图像文件不存在：{message.removeprefix('Image does not exist: ')}"
    if message.startswith("Unsupported image file type: "):
        return f"暂不支持该图像格式：{message.removeprefix('Unsupported image file type: ')}"
    if message.startswith("Unable to read image: "):
        return f"无法读取图像文件：{message.removeprefix('Unable to read image: ')}"
    if message.startswith("Unable to encode image for saving: "):
        return f"无法编码并保存图像：{message.removeprefix('Unable to encode image for saving: ')}"
    if message.startswith("Unexpected image shape: "):
        return f"图像维度不符合预期：{message.removeprefix('Unexpected image shape: ')}"
    if message.startswith("Unexpected runtime error: "):
        return f"运行时出现异常：{message.removeprefix('Unexpected runtime error: ')}"
    return message
