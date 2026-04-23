# BUCAD 中文介绍

BUCAD 是一个面向乳腺超声图像的智能辅助诊断原型系统。项目目标不是替代医生，而是把“上传乳腺超声图像、自动分析风险、标出可疑区域、展示模型关注区域、生成可阅读报告”串成一个可以演示、可以交付、可以继续扩展的软件流程。

## 项目目标

用户输入一张乳腺超声图像后，系统会输出三类结果：

- 良恶性判断：输出恶性概率、良性概率和最终判定结果。
- 病灶区域可视化：通过分割结果或叠加图展示可疑病灶大致位置。
- 辅助解释信息：通过 Grad-CAM 热力图、风险提示和结果说明展示模型关注区域。

整体流程可以概括为：

```text
输入图像 -> 识别风险 -> 展示病灶 -> 输出判断 -> 辅助理解结果
```

## 当前能力

- 已支持 BUSBRA/BUSI 数据读取、病例级划分和泄漏检查。
- 已支持分类模型训练、BUSI 外部评估、阈值分析和指标报告。
- 已加入开发手册推荐方向：EfficientNetV2-S 主模型配置、模型对比配置、Grad-CAM 目标层适配和可视化证据导出。
- 已支持 Gradio 中文操作界面，用户可以上传单张图像并查看诊断结果、病灶叠加图和解释热力图。
- 已支持 PyInstaller 打包和 release 资产导出，便于离线演示。
- 已支持将主要 Markdown/JSON 报告汇总为 Word 文档：`artifacts\reports\documents\bucad_report_summary.docx`。

## 技术栈

- 语言与环境：Python 3.10、Conda
- 深度学习框架：PyTorch、torchvision
- 分类模型库：timm
- 分割模型库：segmentation_models_pytorch
- 图像处理：OpenCV
- 数据处理：NumPy、pandas
- 评估指标：scikit-learn
- 训练日志：TensorBoard、tqdm
- 配置与命令行：PyYAML、argparse
- 可解释性：pytorch-grad-cam
- 操作界面：Gradio
- 打包：PyInstaller
- 版本管理：Git、GitHub

## 快速开始

1. 创建并激活环境：

```powershell
conda create -n BUCAD python=3.10 -y
conda activate BUCAD
pip install -r requirements.txt
```

2. 检查环境：

```powershell
python check_env.py
python check_all.py
```

3. 复制路径配置模板，并改成你的本地数据路径：

```powershell
Copy-Item configs\paths.example.yml configs\paths.local.yml
```

4. 生成病例级安全划分：

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

5. 训练和评估基础分类模型：

```powershell
python scripts\train_cls.py --config configs\classifier\baseline.yml --fold 1 --epochs 1
python scripts\eval_busi.py --config configs\inference\demo.yml
```

6. 运行开发手册主模型配置：

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

7. 运行模型对比证据：

```powershell
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --model-limit 1 --dry-run
python scripts\run_comparison.py --config configs\classifier\comparison.yml --fold 1 --epochs 20
```

8. 启动操作界面：

```powershell
python app\main.py
```

## 报告与交付

项目的正式报告交付格式使用 Word DOCX，不默认生成 PDF。

```powershell
python scripts\export_report_documents.py --reports-dir artifacts\reports --output-dir artifacts\reports\documents
```

生成结果：

```text
artifacts\reports\documents\bucad_report_summary.docx
```

这份 Word 文档会汇总最终验证、模型指标、阈值分析、可视化证据、发布记录和开发手册进度，适合用于归档、答辩准备和队友交接。

## 常用导出命令

导出可视化证据：

```powershell
python scripts\export_visual_evidence.py --config configs\inference\demo.yml --output-dir artifacts\reports\visual_evidence --limit 6
```

批量推理：

```powershell
python scripts\batch_infer.py --config configs\inference\demo.yml --input-dir 测试集\Dataset_BUSI_with_GT\malignant --output artifacts\reports\batch_inference.csv
```

打包演示版本：

```powershell
pyinstaller packaging\demo.spec --noconfirm
python scripts\export_demo_assets.py --config configs\paths.local.yml --output-dir artifacts\release_v1
```

## 项目结构

- `configs/`：路径、分类、分割、推理等 YAML 配置。
- `src/`：数据集、模型、预处理、训练推理、解释性和工具模块。
- `scripts/`：划分、训练、评估、导出和批量推理命令入口。
- `app/`：Gradio 操作界面。
- `tests/`：单元测试、烟雾测试和集成测试。
- `artifacts/`：运行输出、日志、报告和交付资产。
- `specs/001-breast-ultrasound-cad/`：Spec Kit 计划、任务、合同和快速开始文档。

## 当前进度

当前已完成可演示原型、基础训练评估流程、可视化解释、打包流程和 DOCX 报告导出。开发手册进度记录在：

```text
artifacts\reports\handbook_progress.md
```

下一阶段重点是正式实验和最终证据冻结：

- 完成 EfficientNetV2-S 5 折正式训练。
- 完成完整模型对比和 BUSI 最终外部评估。
- 冻结最终模型、阈值和推理配置。
- 整理最终良恶性可视化样例、答辩提纲、常见问答和交接清单。

## 注意事项

- 本项目仅用于辅助诊断研究、比赛展示和原型演示，不能作为临床最终诊断依据。
- 训练数据和测试数据默认不进入 Git 仓库。
- BUSBRA 用于训练和内部验证，BUSI 用于外部评估和演示验证，二者需要保持边界隔离。
- 如果分割权重或解释模块不可用，系统仍应返回基础诊断结果，并在界面中明确提示缺失项。
