# BUCAD 中文说明

BUCAD（Breast Ultrasound Computer-Aided Diagnosis）是一个面向乳腺超声图像的辅助诊断原型项目。它覆盖乳腺超声 CAD 项目的核心工程流程：数据准备、病例级划分、分类模型训练、BUSI 外部评估、病灶可视化、Grad-CAM 风格解释、Gradio 本地演示和发布打包。

> BUCAD 是科研和原型系统，不是临床诊断产品，不能替代医生判断。

## 项目功能

输入一张乳腺超声图像后，系统可以：

- 输出良性/恶性概率；
- 使用可配置阈值给出最终判断；
- 输出置信度和边界样本提示；
- 生成病灶定位叠加图；
- 生成 Grad-CAM 风格解释热力图；
- 对 BUSI 文件夹做批量推理；
- 启动 Gradio 本地演示界面。

## 当前模型

当前运行时分类器是五折 `tf_efficientnetv2_s` ensemble，推理配置位于 `configs/inference/demo.yml`。

运行时设置：

- 分类器：`tf_efficientnetv2_s`
- 权重：`artifacts/checkpoints/efficientnetv2_s_fold1.pt` 到 `efficientnetv2_s_fold5.pt`
- 预处理：启用 CLAHE 和水平翻转 TTA
- 选定阈值：`0.25`
- 分割权重：`artifacts/checkpoints/segmenter_fold1.pt`

## 实验结果

### BUSI 外部评估

当前 EfficientNetV2-S ensemble 在选定阈值 `0.25` 下的 BUSI 外部评估结果：

| 指标 | 数值 |
| --- | ---: |
| AUC | 0.8997 |
| Sensitivity | 0.8476 |
| Specificity | 0.8078 |
| Accuracy | 0.8207 |

评估脚本仍会在 `artifacts/reports/busi_tta_eval.json` 顶层 `metrics` 中保留传统 `0.50` 阈值指标；当前采用的运行点在 `threshold_analysis.best_by_youden` 中。

### 模型对比

当前记录的 fold-1、20 epoch 对比实验：

| 排名 | 模型 | AUC | Sensitivity | Specificity | 状态 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `tf_efficientnetv2_s` | 0.8937 | 0.7049 | 0.8775 | completed |
| 2 | `densenet121` | 0.8867 | 0.7213 | 0.8775 | completed |
| 3 | `resnet18` | 0.8756 | 0.7131 | 0.8577 | completed |
| 4 | `mobilenetv3_small_100` | 0.8704 | 0.6557 | 0.9170 | completed |
| 5 | `basic_cnn` | 0.6427 | 0.0000 | 0.9921 | completed |
| 6 | `vgg16` | 0.5000 | 0.0000 | 1.0000 | completed |
| - | `alexnet` | - | - | - | recorded run failed |

`alexnet` 支持是在这次记录之后补上的。如果需要更新后的 AlexNet 指标，需要重跑对比实验。

## 项目目录

- `configs/`：路径、分类器、分割器和推理配置。
- `src/datasets/`：BUSBRA/BUSI 数据集读取和 split 支持。
- `src/models/`：分类器和分割器工厂。
- `src/engine/`：训练、对比、推理和评估流程。
- `src/explain/`：Grad-CAM 和可视化叠加图。
- `src/preprocess/`：图像读取和预处理。
- `src/utils/`：配置、指标、报告、路径、日志和结果结构。
- `scripts/`：命令行脚本。
- `app/`：Gradio 应用。
- `tests/`：单元测试和 smoke 测试。
- `artifacts/`：本地产物、权重、报告和 release 包。

## 数据路径

本地路径在 `configs/paths.local.yml` 中配置：

```yaml
datasets:
  busbra_root: ./训练集/BUSBRA
  busi_root: ./测试集/Dataset_BUSI_with_GT
```

数据边界：

- BUSBRA 用于训练和内部验证。
- BUSI 用于外部评估和演示验证。
- split 使用病例级划分，避免数据泄漏。
- 数据集、权重、生成图片、JSON/CSV 报告和 release 包默认不进入 Git。

## 环境安装

```powershell
conda create -n BUCAD python=3.11 -y
conda activate BUCAD
python -m pip install -r requirements.txt
```

检查环境和测试：

```powershell
python check_env.py
python check_all.py
```

## 生成数据划分

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

输出：

- `artifacts/reports/busbra_5fold_splits.csv`
- `artifacts/reports/busbra_split_summary.json`

## 训练分类模型

训练单折：

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

训练五折：

```powershell
scripts\train_all_folds.bat
```

输出：

- `artifacts/checkpoints/efficientnetv2_s_fold1.pt` 到 `efficientnetv2_s_fold5.pt`
- `artifacts/reports/train_cls_efficientnetv2_s_fold1.json` 到 `fold5.json`

## 运行模型对比

快速 dry-run：

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --model-limit 1 --dry-run
```

完整对比：

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
```

输出：

- `artifacts/reports/comparison_results.json`
- `artifacts/reports/comparison_summary.md`
- 各模型权重：`artifacts/checkpoints/`
- 各模型报告：`artifacts/reports/comparison_*_fold1.json`

## BUSI 外部评估

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_tta_eval.json
```

输出：

- `artifacts/reports/busi_tta_eval.json`
- `artifacts/reports/threshold_analysis.md`

## 启动演示界面

```powershell
python app\main.py
```

该命令会启动本地 Gradio 界面，用于单图诊断和可视化查看。

## 导出项目产物

导出可视化证据：

```powershell
python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence_final --limit 6
```

导出批量推理：

```powershell
python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference_final.csv
```

导出 Word 报告摘要：

```powershell
python scripts\export_report_documents.py --reports-dir artifacts\reports --output-dir artifacts\reports\documents
```

打包 release：

```powershell
pyinstaller packaging\demo.spec --noconfirm
python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
```

## 常用报告

- `artifacts/reports/efficientnetv2_s_5fold_summary.md`
- `artifacts/reports/model_freeze_decision.md`
- `artifacts/reports/comparison_summary.md`
- `artifacts/reports/report_tables.md`
- `artifacts/reports/resolution_augmentation_experiment.md`
- `artifacts/reports/final_validation.md`
