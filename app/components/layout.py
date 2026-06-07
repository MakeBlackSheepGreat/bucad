from __future__ import annotations


EMPTY_DIAGNOSIS_HTML = """
<div class="result-card">
  <div class="section-title">诊断结果</div>
  <div class="empty-state">上传图像并点击“开始诊断”后显示结果。</div>
</div>
"""

EMPTY_STATUS_HTML = """
<div class="status-card">
  <div class="section-title">运行状态</div>
  <div class="hint-text">等待上传图像。</div>
</div>
"""

EMPTY_WARNING_HTML = """
<div class="status-card">
  <div class="section-title">辅助说明</div>
  <div class="hint-text">本系统仅用于辅助分析和原型演示，不能替代医生诊断。</div>
</div>
"""


def hero_html(ensemble_display_name: str) -> str:
    return f"""
<div class="hero">
  <div class="logo-mark">⌁</div>
  <div>
    <h1>乳腺超声肿瘤良恶性分类辅助诊断系统（BUCAD）</h1>
    <p>Breast Ultrasound Computer-Aided Diagnosis · {ensemble_display_name} · ROI 定位 · Grad-CAM 解释</p>
  </div>
</div>
"""


def viewer_header_html() -> str:
    return """
<div class="viewer-title">
  <span>图像分析结果</span>
  <span>原图 / 分割 / 热力图</span>
</div>
"""


def footer_note_html() -> str:
    return '<div class="footer-note">提示：分割图和热力图用于辅助理解模型输出，不代表临床标注或最终诊断。</div>'
