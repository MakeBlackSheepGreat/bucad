<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# Demo 彩排检查清单

日期：2026-04-24

状态：自动化和打包冒烟证据已经记录；最终现场展示前仍建议进行人工彩排。

- [x] 确认自动化 Gradio smoke test 通过。
- [x] 导出最终 BUSI 可视化证据样本。
- [x] 运行最终 BUSI batch inference 导出。
- [x] 确认 lesion overlay 存在，或缺失原因可见。
- [x] 确认 Grad-CAM heatmap 存在，或缺失原因可见。
- [x] 确认 packaged assets 位于 `artifacts/release_v1/`。
- [x] 确认 `artifacts/reports/release_v1_manifest.md` 可用于交接验证。
- [x] 确认打包 exe 能启动并在 20 秒 smoke window 内保持运行。
- [ ] 为 PPT 截取最终浏览器实时截图。
- [ ] 在实时浏览器彩排中确认辅助用途免责声明可见。

中文 UI 彩排提示：把系统介绍为辅助诊断 prototype；先解释概率输出，再解释 lesion overlay，最后解释 Grad-CAM heatmap；必须明确医学诊断仍由医生主导。
