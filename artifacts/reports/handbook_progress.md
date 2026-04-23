# 开发手册进度同步

日期：2026-04-23

## 同步依据

- 核心手册：`开发手册_V2.md`
- 当前任务清单：`specs/001-breast-ultrasound-cad/tasks.md`
- 当前计划：`specs/001-breast-ultrasound-cad/plan.md`

## 已完成

- 项目基础能力已完成：环境检查、配置加载、BUSBRA/BUSI 数据读取、病例级划分、训练/评估脚本、指标报告、Gradio 演示界面和打包流程。
- 开发手册模型方向已落地：已加入 EfficientNetV2-S 主模型配置、对比实验配置、阈值分析、Grad-CAM 目标层适配、可视化证据导出和批量推理导出。
- 已完成 Phase 7 到 Phase 10 的手册跟进任务：模型质量证据、可视化证据强化、发布准备、阶段性证据冻结。
- 报告交付格式已调整为 Word DOCX：`scripts\export_report_documents.py` 现在默认只生成 `artifacts\reports\documents\bucad_report_summary.docx`。

## 待完成

- Phase 11：正式模型证据。需要完成正式实验前置检查、5 折 EfficientNetV2-S 训练、完整模型对比、BUSI 最终外部评估和最终模型/阈值冻结说明。
- Phase 12：正式可解释性证据。需要整理最终良恶性样例、病灶叠加图、Grad-CAM 热力图和人工审阅记录。
- Phase 13：发布候选冻结。需要校验最终推理配置、导出最终 release 资产、完成打包演示验证和中文 UI 彩排。
- Phase 14：报告与答辩包。需要完成最终表格、答辩提纲、常见问答、交接清单和最终验证记录。

## 下一步建议

- 先执行 T057 和 T058，确认数据、权重目录、命令清单和实验预算。
- 再执行 T059 到 T062，优先产出正式 EfficientNetV2-S 5 折和 BUSI 最终评估证据。
- 在最终模型冻结后再推进 Phase 12 到 Phase 14，避免视觉证据和报告结论反复改动。
