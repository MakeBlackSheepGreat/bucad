<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 最终交接检查清单

日期：2026-04-24

## 命令

- 环境检查：`conda run -n BUCAD python check_env.py`
- 回归检查：`conda run -n BUCAD python check_all.py`
- BUSI 最终评估：`conda run -n BUCAD python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_eval_final.json`

## 关键产物

- `configs/inference/demo.yml`
- `artifacts/reports/busi_eval_final.json`
- `artifacts/reports/visual_evidence_final/`
- `artifacts/reports/documents/bucad_report_summary.docx`
- `artifacts/release_v1/`

## 风险

- 数据集、checkpoint 和 release bundle 不应提交到 Git。
- BUSI 只能用于外部评估，不能用于训练或调参。
- 最终展示前需要人工检查可视化案例是否合理。

## 建议负责人

- 模型评估负责人：复核 BUSI 指标和阈值。
- Demo 负责人：复核 Gradio 页面和打包程序。
- 文档负责人：同步 README、答辩文稿和 DOCX 报告。
