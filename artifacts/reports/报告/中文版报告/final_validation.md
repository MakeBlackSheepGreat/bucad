<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 最终验证状态

日期：2026-04-23

## 环境

- 验证环境：Conda `BUCAD`
- Python：`3.11.15`
- OS：`Windows 10`
- CUDA available：`True`
- GPU：`NVIDIA GeForce RTX 5060 Laptop GPU`

说明：plan 和 quickstart 最初以 Python 3.10 为目标，但本次最终验证按照 2026-04-23 的本地环境决策在 Python 3.11.15 上完成。

## Quickstart 验证

以下命令在当前环境中成功执行：

- `python check_env.py`
- `python check_all.py`
- `python scripts\make_split.py --config configs\paths.local.yml`
- `python scripts\train_cls.py --config configs\classifier\baseline.yml --fold 1 --epochs 1`
- `python scripts\train_seg.py --config configs\segmenter\unet.yml --fold 1 --epochs 1`
- `python scripts\eval_busi.py --config configs\inference\demo.yml`
- Source Gradio smoke：`build_app().launch(prevent_thread_lock=True, server_name='127.0.0.1', server_port=7861, share=False)` 后可正常关闭。
- `pyinstaller packaging\demo.spec --noconfirm`
- `python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir dist\bucad-demo`
- Packaged demo smoke：`dist\bucad-demo\bucad-demo.exe` 在导出 runtime configs 和 checkpoints 后保持运行 20 秒，无 stderr。

## 自动化回归状态

- DOCX-only report export 同步后的完整测试：`32 passed in 10.37s`

## 指标与输出

- BUSBRA split：`1875` samples、`1064` unique cases、`5` folds、`leakage_detected=false`
- Classifier fold 1 smoke：checkpoint `artifacts/checkpoints/classifier_fold1.pt`，AUC `0.8062`，Sensitivity `0.4098`，Specificity `0.9051`，Accuracy `0.7440`
- Segmenter fold 1 smoke：checkpoint `artifacts/checkpoints/segmenter_fold1.pt`，Dice `0.8100`
- BUSI 外部评估：`artifacts/reports/busi_eval.json`，sample count `647`，AUC `0.7564`，Sensitivity `0.6095`，Specificity `0.8009`，Accuracy `0.7388`
- 单图 demo 证据：输入为本地 BUSI malignant 样本，输出包括 `diagnosis.json`、`original.png`、`lesion_overlay.png` 和 `explanation.png`

## 观察

- 端到端系统已能完成分类、lesion overlay、explanation overlay、Gradio 启动和打包 demo 启动。
- 单图 malignant 样本在 smoke run 中被判为 benign，恶性概率 `0.3769`；这说明 smoke 通过不等于临床可靠，系统必须定位为辅助 demo。
- 打包流程需要把 runtime assets 导出到 `dist\bucad-demo`，以便 exe 能解析 `configs/inference/demo.yml` 和 checkpoint。

## 后续验证记录

- 2026-04-23：模型流水线后续实现完成，回归 `30 passed in 9.28s`。
- 2026-04-23：DOCX 报告导出完成，`tests\integration\test_report_documents.py` 为 `2 passed`，完整回归 `32 passed in 10.37s`。
- 2026-04-24：Release candidate follow-up 完成，`check_all.py` 为 `26 passed in 9.29s`，导出最终可视化证据、BUSI eval、batch inference、release assets 和 packaged demo smoke。
- 2026-04-24：EfficientNetV2-S T059 五折完成，mean AUC `0.8946`，BUSI ensemble AUC `0.8955`，Youden threshold `0.25`。
- 2026-04-24：T060 完整对比完成，7 个配置模型中 `tf_efficientnetv2_s` 以 AUC `0.8937` 排名第一；`alexnet` 在记录运行中失败。
- 2026-04-24：修复 comparison dry-run 隔离、加入 AlexNet 构建支持、更新 Windows pytest temp 目录策略；`check_all.py` 完成 `26 passed in 9.74s`。
