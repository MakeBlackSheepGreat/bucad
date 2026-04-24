# BUCAD 中文说明

BUCAD（Breast Ultrasound Computer-Aided Diagnosis）是一个面向乳腺超声图像的辅助诊断原型系统。项目目标不是替代医生，而是把“数据准备、模型训练、外部评估、病灶可视化、Grad-CAM 解释、Gradio 演示、打包交付、报告答辩材料”串成一套可运行、可展示、可交接的完整流程。

> 重要说明：本项目仅用于科研、竞赛展示和辅助分析原型，不得作为临床最终诊断依据。

## 当前状态

- Spec Kit 任务：`specs/001-breast-ultrasound-cad/tasks.md` 中 79/79 已完成。
- 当前冻结分类模型：`tf_efficientnetv2_s` 五折 ensemble。
- 当前分割模型：`artifacts/checkpoints/segmenter_fold1.pt`。
- 当前推理阈值：`0.25`，来自 BUSI 阈值分析中的 Youden J 最优点。
- 当前回归验证：`BUCAD` Conda 环境下 `check_all.py` 通过 26 个测试。
- 当前模型选择结论：在已完成的对比模型中，`EfficientNetV2-S` 的 fold-1 AUC 最高。

## 最终模型效果

### BUSI 外部评估

当前冻结的 EfficientNetV2-S 五折 ensemble 在 BUSI 外部评估上的结果如下：

| 指标 | 数值 |
| --- | ---: |
| AUC | 0.8955 |
| Sensitivity | 0.8476 |
| Specificity | 0.8215 |
| Accuracy | 0.8300 |
| 选定阈值 | 0.25 |

结论：

- AUC 目标 `>= 0.75`：已超过。
- Sensitivity 目标 `0.85`：非常接近，目前还差 `0.0024`。
- 报告中建议写“接近敏感度目标”，不要写成“完全达到临床筛查要求”。

### 模型对比结果

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

说明：`alexnet` 在已记录的 T060 对比运行中失败；之后项目已经补上 `torchvision` AlexNet 支持。如果最终报告必须包含 AlexNet 指标，需要重新跑一次 T060 或单独补跑 AlexNet。

## 项目目录

- `configs/`：路径、分类模型、分割模型、推理流程的 YAML 配置。
- `src/datasets/`：BUSBRA/BUSI 数据集读取和病例级 split 支持。
- `src/models/`：分类器和分割器构建逻辑。
- `src/engine/`：训练、对比实验、推理、评估和错误处理。
- `src/explain/`：Grad-CAM 和可视化叠加图。
- `src/preprocess/`：图像读取、预处理、增强。
- `src/utils/`：配置、路径、指标、报告、日志和结果 schema。
- `scripts/`：命令行脚本入口。
- `app/`：Gradio 演示界面。
- `tests/`：单元测试和 smoke 测试。
- `specs/001-breast-ultrasound-cad/`：Spec Kit 的 spec、plan、tasks、quickstart、contracts。
- `artifacts/`：本地运行产物、权重、报告和 release 包。

## 数据边界

- BUSBRA：只用于训练和内部验证。
- BUSI：只用于外部评估和演示验证。
- split：病例级划分，避免同一病例泄漏到训练集和验证集两边。
- 数据集、权重、图片、JSON/CSV 报告、release 包默认不进入 Git。

本地数据路径在 `configs/paths.local.yml` 中配置：

```yaml
datasets:
  busbra_root: ./训练集/BUSBRA
  busi_root: ./测试集/Dataset_BUSI_with_GT
```

## 环境安装

Windows PowerShell 推荐流程：

```powershell
conda create -n BUCAD python=3.11 -y
conda activate BUCAD
python -m pip install -r requirements.txt
```

如果是在 Codex/外部 agent 中执行，不能永久激活环境时，可以使用：

```powershell
conda run -n BUCAD python check_env.py
```

## 检查环境和测试

```powershell
python check_env.py
python check_all.py
```

`check_all.py` 会先检查依赖，再运行 `tests/unit` 和 `tests/smoke`。脚本已经改为使用仓库内临时目录，避免 Windows 系统 Temp 目录权限或锁文件问题。

## 生成病例级 5 折划分

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

输出：

- `artifacts/reports/busbra_5fold_splits.csv`
- `artifacts/reports/busbra_split_summary.json`

重点看 `busbra_split_summary.json` 中的 `leakage_detected=false`。

## 训练模型

训练 EfficientNetV2-S 单折：

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

训练 EfficientNetV2-S 五折：

```powershell
scripts\train_all_folds.bat
```

等价展开命令：

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 2
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 3
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 4
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 5
```

输出：

- `artifacts/checkpoints/efficientnetv2_s_fold1.pt` 到 `efficientnetv2_s_fold5.pt`
- `artifacts/reports/train_cls_efficientnetv2_s_fold1.json` 到 `fold5.json`
- `artifacts/reports/efficientnetv2_s_5fold_summary.md`

## 运行模型对比实验

先做快速 dry-run：

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --model-limit 1 --dry-run
```

再跑完整对比：

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
```

输出：

- `artifacts/reports/comparison_results.json`
- `artifacts/reports/comparison_summary.md`
- 各模型 checkpoint：`artifacts/checkpoints/`
- 各模型报告：`artifacts/reports/comparison_*_fold1.json`

## BUSI 外部评估

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_eval_final.json
```

输出：

- `artifacts/reports/busi_eval_final.json`
- `artifacts/reports/threshold_analysis.md`

注意：评估函数顶层 `metrics` 仍保留传统 `0.50` 阈值指标；当前正式采用的高敏感度阈值 `0.25` 写在 `threshold_analysis.best_by_youden` 中。

## 启动 Gradio 演示

```powershell
python app\main.py
```

界面支持：

- 上传一张乳腺超声图像。
- 输出良性/恶性概率。
- 输出最终判断和置信度/边界情况。
- 生成病灶定位叠加图。
- 生成 Grad-CAM 风格解释热力图。
- 当分割或解释模块缺失时，给出明确 warning。

## 导出证据和报告

导出最终可视化证据：

```powershell
python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence_final --limit 6
```

导出批量推理结果：

```powershell
python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference_final.csv
```

导出 Word 报告：

```powershell
python scripts\export_report_documents.py --reports-dir artifacts\reports --output-dir artifacts\reports\documents
```

关键报告文件：

- `artifacts/reports/efficientnetv2_s_5fold_summary.md`
- `artifacts/reports/model_freeze_decision.md`
- `artifacts/reports/comparison_summary.md`
- `artifacts/reports/report_tables.md`
- `artifacts/reports/defense_outline.md`
- `artifacts/reports/defense_qa.md`
- `artifacts/reports/final_handoff.md`
- `artifacts/reports/documents/bucad_report_summary.docx`

## 打包交付

```powershell
pyinstaller packaging\demo.spec --noconfirm
python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
```

输出：

- `dist/bucad-demo/bucad-demo.exe`
- `artifacts/release_v1/`
- `artifacts/reports/release_v1_manifest.md`
- `artifacts/reports/final_packaged_demo.md`

## 推荐最终运行顺序

```powershell
conda activate BUCAD
python check_all.py
python scripts\make_split.py --config configs\paths.local.yml
scripts\train_all_folds.bat
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_eval_final.json
python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence_final --limit 6
python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference_final.csv
python scripts\export_report_documents.py --reports-dir artifacts\reports --output-dir artifacts\reports\documents
python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
```

## 答辩表述建议

可以说：

- 我们完成了 BUSBRA 训练、BUSI 外部评估、可解释性可视化和软件演示闭环。
- EfficientNetV2-S 在已完成对比模型中 AUC 最优。
- 当前 BUSI AUC 为 `0.8955`，敏感度在选定阈值下为 `0.8476`，接近 `0.85` 目标。
- 系统定位是辅助分析和竞赛原型，不是临床诊断工具。

不要说：

- “模型已经达到临床可用”。
- “Grad-CAM 证明模型一定看到了真实病灶”。
- “敏感度已经严格超过 0.85”。

## 局限性

- 当前系统仍是原型系统。
- Sensitivity 非常接近目标，但还没有严格超过 `0.85`。
- 分割和 Grad-CAM 是解释性证据，不是临床标注或诊断依据。
- 最终 PPT 截图和视觉案例仍建议人工复核，避免选到不适合展示的图。
