<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 答辩提纲

日期：2026-04-24

## 1. 问题与目标

- 任务：面向乳腺超声图像的良恶性辅助诊断。
- 交付物：可运行的 CAD 原型，包含分类、病灶定位、解释热力图和 Gradio demo。
- 边界：仅用于辅助分析，不替代临床诊断。

## 2. 数据治理

- BUSBRA 用于训练和内部验证。
- BUSI 保留为外部评估和 demo 验证。
- 使用病例级 split，避免同一病例跨 fold 泄漏。

## 3. 方法

- 分类流程：预处理、模型推理、概率输出、基于阈值的判定、置信度/边界状态处理。
- 定位流程：用 segmentation overlay 展示病灶区域。
- 可解释流程：用 Grad-CAM 风格热力图检查模型关注区域。
- Demo 流程：Gradio 上传、结果面板、warning/fallback 信息和 release bundle。

## 4. 证据

- BUCAD 环境下回归测试通过。
- BUSI 最终评估：见 `artifacts/reports/busi_eval_final.json`。
- 可视化证据：见 `artifacts/reports/visual_evidence_final/README.md`。
- Release 清单：见 `artifacts/reports/release_v1_manifest.md`。

## 5. 局限与后续工作

- 当前冻结模型是 release-candidate prototype；正式 EfficientNetV2-S 五折证据当时仍待补充。
- BUSI Sensitivity 低于临床筛查目标。
- 后续工作：完成完整模型对比、提升 Sensitivity、加入临床视觉复核、校准阈值。
## 比赛指标完整性说明

比赛统一指标集为 AUC、Accuracy、Recall/Sensitivity、Precision、Specificity 和 F1-Score。如果历史归档表中 Precision 或 F1-Score 显示 `-`，说明原始摘要未保留可反推该值的 confusion matrix 或原始概率。锁定 BUSI 运行点结果以 `artifacts/reports/competition_metrics_all_busi_reports.md` 作为完整指标来源。

