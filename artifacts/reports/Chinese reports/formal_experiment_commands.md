<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 正式实验命令

日期：2026-04-24

## 执行规则

- 所有命令都从仓库根目录运行：`C:\Users\876762330\Desktop\Agent`。
- 使用现有 Conda 环境：`conda run -n BUCAD ...`。
- BUSBRA 只用于训练和内部验证。
- BUSI 只用于外部评估和 demo 验证。
- 不提交数据集、checkpoint、release bundle 或生成的大型 artifact。

## 预检查

```powershell
conda run -n BUCAD python check_env.py
conda run -n BUCAD python check_all.py
conda run -n BUCAD python scripts\make_split.py --config configs\paths.local.yml
```

期望证据：

- `check_env.py` 报告 required 和 optional packages 可用。
- `check_all.py` 通过 smoke/unit suite。
- `artifacts/reports/busbra_5fold_splits.csv` 存在，且 `artifacts/reports/busbra_split_summary.json` 中 `leakage_detected=false`。

## T059：EfficientNetV2-S 五折训练

```powershell
conda run -n BUCAD python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
conda run -n BUCAD python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 2
conda run -n BUCAD python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 3
conda run -n BUCAD python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 4
conda run -n BUCAD python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 5
```

期望证据包括五个 checkpoint 和五个 `train_cls_efficientnetv2_s_fold*.json` 报告。

## T060：完整模型对比

```powershell
conda run -n BUCAD python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
```

期望证据：`artifacts/reports/comparison_results.json`，每个模型结果应为 `status=completed` 或有明确失败原因。

## T061：最终 BUSI 外部评估

最终分类 checkpoint 或 ensemble 冻结后，更新 `configs/inference/demo.yml`，再运行：

```powershell
conda run -n BUCAD python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_eval_final.json
```

期望证据：`artifacts/reports/busi_eval_final.json`，包含 AUC、Sensitivity、Specificity、Accuracy、confusion matrix 和 threshold analysis。

## T062：冻结决策

T059-T061 完成后创建 `artifacts/reports/model_freeze_decision.md`，必须说明选定 classifier、EfficientNetV2-S fold metrics、完整对比、BUSI 外部评估、最终阈值、置信度/边界处理、已知局限和恢复方案。

## 后续 Release 命令

```powershell
conda run -n BUCAD python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence_final --limit 6
conda run -n BUCAD python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference_final.csv
conda run -n BUCAD python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
conda run -n BUCAD python scripts\export_report_documents.py --reports-dir artifacts\reports --output-dir artifacts\reports\documents
```

期望证据：`visual_evidence_final/README.md`、`batch_inference_final.csv`、`release_v1_manifest.md` 和 `bucad_report_summary.docx`。
## 比赛指标完整性说明

比赛统一指标集为 AUC、Accuracy、Recall/Sensitivity、Precision、Specificity 和 F1-Score。如果历史归档表中 Precision 或 F1-Score 显示 `-`，说明原始摘要未保留可反推该值的 confusion matrix 或原始概率。锁定 BUSI 运行点结果以 `artifacts/reports/competition_metrics_all_busi_reports.md` 作为完整指标来源。
