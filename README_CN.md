# BUCAD - 乳腺超声肿瘤分类与分割辅助系统

BUCAD（Breast Ultrasound Computer-Aided Diagnosis）是一个面向乳腺超声影像的计算机辅助诊断原型系统。项目围绕“良恶性分类 + 病灶区域辅助定位 + 可解释性展示”构建，将分类模型、分割模型、ROI 引导推理、Grad-CAM 热力图、本地 Gradio 演示界面和 Windows 桌面打包整合在同一个 Python 工程中。

项目适合用于医学影像算法验证、教学演示、科研原型复现和后续二次开发。第三方使用者可以直接运行 demo 查看单张图像推理结果，也可以基于现有训练、评估和报告脚本继续扩展新的模型、数据集或推理策略。

> BUCAD 是科研和原型系统，不是临床诊断产品，不能替代医生判断。

## 项目功能

输入一张乳腺超声图像后，系统可以：

- 输出良性/恶性概率；
- 根据可配置阈值给出最终判定；
- 输出置信度和边界样本提醒；
- 在分割权重可用时生成病灶定位叠加图；
- 从主分类模型生成 Grad-CAM 风格热力图；
- 对有标签数据集目录进行可复现批量评估；
- 启动本地 Gradio 网页界面进行单图演示。

系统的典型工作流如下：

1. 读取输入图像并完成灰度图处理、CLAHE 增强、尺寸缩放和模型专属归一化。
2. 使用五折分类模型分别预测完整图像的良恶性概率。
3. 使用分割模型生成病灶 mask，并根据 mask 裁剪 ROI 区域。
4. 使用同一组分类模型再次评估 ROI 区域，得到局部病灶视角下的恶性概率。
5. 通过 OOF 训练得到的轻量逻辑融合器整合完整图概率和 ROI 概率。
6. 根据默认阈值输出最终分类结果、置信度、分割叠加图和 Grad-CAM 热力图。

## 当前演示模型

当前主线是 `ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate`，运行配置在 `configs/inference/demo.yml`。

- **主模型分支**：`ConvNeXt-Tiny` 五折 checkpoint，集成权重 `0.573`，使用 timm-aware 预处理和 crop-sweep TTA；网页 UI 与 Grad-CAM 解释也以它作为主模型。
- **辅助模型分支**：`EfficientNetV2-S` 五折 checkpoint，集成权重 `0.427`，使用 CLAHE 预处理和 identity TTA；它用于补充 ConvNeXt-Tiny 的错误分布，同时比继续加入 DenseNet/Swin 更轻量。
- **ROI 分支**：`segmenter_fold1.pt` 预测病灶 mask；根据训练集分割验证结果使用 `mask_threshold=0.40`、`margin_ratio=0.35` 和最大连通域裁剪。
- **OOF 融合**：系统会同时计算完整图概率和 ROI 概率，再输入训练集 out-of-fold 预测训练出的 logit logistic stacker 得到最终恶性概率。
- **ROI 面积质量门控**：ROI 面积比例低于 `0.08` 或高于 `0.75` 时回退到完整图预测，降低异常 ROI 对最终分类的影响。
- **判定阈值**：`0.510`，作为打包 demo 的默认运行点。

当前 demo 配置的代表性评估结果：

| 模型 | 阈值 | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF Stacking + LCC ROI + Area Gate | 0.510 | 0.9256 | 0.8532 | 0.8667 | 0.7309 | 0.8467 | 0.7930 |

可选保留配置：

| 模型 | 阈值 | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny + EfficientNetV2-S + DenseNet121 非 ROI 优化集成 | 0.453 | 0.9162 | 0.8702 | 0.7476 | 0.8351 | 0.9291 | 0.7889 | 可选高特异性对比配置 |
| ConvNeXt-Tiny + EfficientNetV2-S + DenseNet121 + ROI OOF LCC | 0.560 | 0.9229 | 0.8501 | 0.8143 | 0.7467 | 0.8673 | 0.7790 | 可选三模型 ROI 线 |

这些更重的三模型方案作为可选配置保留，便于复现实验和横向比较。默认 demo 采用两模型 ROI Area Gate 主线，因为它在当前评估中具有更高的 AUC、Recall/Sensitivity 和 F1-Score，同时部署复杂度低于 15 个 checkpoint 的三模型集成。
项目中的模型结果表统一采用 AUC、Accuracy、Recall/Sensitivity、Precision、Specificity 和 F1-Score，便于不同配置之间进行一致比较。

## 模型设计思路

当前默认配置选择两模型 ROI 主线，而不是直接使用更大的三模型集成，主要基于以下工程取舍：

- **ConvNeXt-Tiny 做主模型**：它在轻量级模型中具有较强的图像特征提取能力，并且适合配合 timm 推荐预处理、crop-sweep TTA 和 Grad-CAM 可解释性输出。
- **EfficientNetV2-S 做辅助分支**：它与 ConvNeXt-Tiny 的错误分布不同，能够在不显著增加部署复杂度的情况下改善集成稳定性。
- **ROI 引导降低背景干扰**：完整图像保留全局组织结构，ROI 裁剪强调病灶区域；二者融合能兼顾全局上下文和局部病灶细节。
- **面积质量门控增强鲁棒性**：当分割 mask 过小或过大时，ROI 裁剪往往不稳定，系统会回退到完整图预测，减少异常 ROI 对最终结果的影响。
- **DenseNet121 保留为可选配置**：三模型方案在 Precision 或 Specificity 上有参考价值，但需要更多 checkpoint，且当前默认运行点下 Sensitivity 不如两模型主线。
- **Swin-Tiny 不进入默认 demo**：混合集成测试中它对当前最优运行点贡献有限，因此不纳入默认部署配置。

## 推理与优化流程

项目当前推理流程围绕“稳定、可解释、便于部署”设计：

1. **异构预处理**：每个集成成员可以独立设置图像尺寸、CLAHE、归一化、插值、crop 比例和 TTA。
2. **ConvNeXt crop-sweep TTA**：ConvNeXt 使用 ImageNet mean/std、bicubic 插值和 `crop_pct=0.90/0.95/1.00`，每个 crop 同时评估原图和水平翻转。
3. **EfficientNet identity 分支**：EfficientNetV2-S 使用 CLAHE、area 插值、不继承 ConvNeXt 的 mean/std，并只保留 identity TTA，避免跨模型预处理串扰。
4. **ROI OOF Stacking**：最终 demo 同时使用完整图概率和 ROI 概率，由 OOF 训练出的逻辑融合器输出最终概率。
5. **ROI mask 后处理**：根据分割验证结果，最终 ROI 使用 `0.40` mask 阈值、最大连通域裁剪和 `margin_ratio=0.35`。
6. **ROI 面积质量门控**：极小或过大的 ROI 裁剪会回退到完整图预测，提高当前运行点下的 Precision/F1。
7. **OOF 阈值选择**：最终 ROI Area Gate 阈值为 `0.510`，来自基于训练集 out-of-fold 证据筛选出的候选配置。
8. **可解释性对齐**：`demo.yml` 将 ConvNeXt-Tiny 放在第一分支，网页 UI 和 Grad-CAM 输出都围绕主模型解释，便于第三方理解结果来源。

## 指标说明

项目报告和 README 中的模型基准至少包含以下指标：

- **AUC**：衡量模型区分良恶性样本的整体排序能力。
- **Accuracy**：所有样本中预测正确的比例。
- **Recall/Sensitivity**：实际恶性样本中被正确识别为恶性的比例，反映漏诊风险。
- **Precision**：预测为恶性的样本中真实恶性的比例，反映阳性预测可靠性。
- **Specificity**：实际良性样本中被正确识别为良性的比例，反映误诊控制能力。
- **F1-Score**：Precision 与 Recall/Sensitivity 的调和平均，用于综合评价恶性检出质量。

## 项目目录

- `configs/`：路径、分类器、分割器和推理配置。
- `src/datasets/`：数据集读取与 split 支持。
- `src/models/`：分类器和分割器工厂。
- `src/engine/`：训练、对比、推理和评估流程。
- `src/explain/`：Grad-CAM 和可视化叠加图。
- `src/preprocess/`：图像读取和预处理。
- `src/utils/`：配置、指标、报告、路径、日志和结果结构。
- `scripts/`：命令行入口。
- `app/`：Gradio 网页应用。
- `tests/`：单元、集成和 smoke 测试。
- `artifacts/reports/`：可版本化的 Markdown 实验报告。
- `artifacts/checkpoints/`：本地模型权重，默认不进入 Git。

## 数据路径

从 `configs/paths.example.yml` 复制生成 `configs/paths.local.yml`，并配置本地数据集路径：

```yaml
datasets:
  busbra_root: ./BUSBRA
  busi_root: ./Dataset_BUSI_with_GT
```

数据集说明：

- 按照实际使用场景配置本地数据集路径；目录名称可以根据本机数据组织方式调整。
- split 使用病例级划分，避免数据泄漏。
- 数据集、模型权重、生成图片、JSON/CSV 输出和 release 包默认不进入 Git。

## 环境安装

```powershell
conda create -n BUCAD python=3.11 -y
conda activate BUCAD
python -m pip install -r requirements.txt
```

检查环境：

```powershell
python check_env.py
python check_all.py
```

## 硬件与软件要求

### 运行 demo / 推理

- **操作系统**：主要面向 Windows 10/11 x64。源码运行方式也可以在 PyTorch、OpenCV 等依赖可用的标准 Python 环境中运行。
- **打包桌面版 demo**：解压 release 包后不需要本机额外安装 Python。桌面版使用 Microsoft Edge WebView2，大多数 Windows 10/11 已内置；如果机器缺少该组件，需要安装官方 WebView2 Runtime。
- **CPU / GPU**：支持 CPU 推理，普通演示场景可以使用 CPU。NVIDIA GPU 不是必须项，主要用于降低推理等待时间。
- **内存**：最低建议 8 GB RAM，推荐 16 GB RAM，以保证启动、图像显示和可视化结果更稳定。
- **磁盘空间**：建议至少预留 8 GB，用于解压后的 demo、模型权重、临时文件和生成的可视化结果。

### 训练 / 实验

- **Python 环境**：推荐 Conda + Python 3.10 或 3.11；示例安装命令使用 `BUCAD` 环境和 Python 3.11。
- **GPU**：强烈建议使用 NVIDIA CUDA GPU。CPU 只适合做 smoke test，不适合完整五折训练。
- **显存**：当前 224 分辨率的 ConvNeXt-Tiny / EfficientNetV2-S 实验，8 GB VRAM 可以作为较低可用门槛；如果要更快地跑五折、更大 batch size 或更高分辨率实验，推荐 12-16 GB 或更高显存。
- **系统内存**：最低建议 16 GB RAM，推荐 32 GB RAM，用于训练、报告生成和数据加载。
- **磁盘空间**：建议至少预留 50 GB，用于本地数据集、五折 checkpoint、日志、OOF 产物、报告和临时构建文件。
- **数据处理**：数据集、checkpoint、生成报告和 release 包应保留在 Git 跟踪源码之外。

## 启动网页 demo

浏览器模式：

```powershell
conda activate BUCAD
python app\main.py
```

桌面窗口模式：

```powershell
conda activate BUCAD
python app\desktop_main.py
```

两种模式都会读取 `configs/inference/demo.yml`。当前 demo 声明 `ConvNeXt-Tiny` 为主模型，`EfficientNetV2-S` 为辅助集成分支。

## Windows 一键启动 Demo

自动打开浏览器的版本：

```powershell
conda activate BUCAD
python -m PyInstaller --clean --noconfirm packaging\demo.spec
```

桌面窗口版本：

```powershell
conda activate BUCAD
python -m PyInstaller --clean --noconfirm packaging\desktop_demo.spec
```

- 自动打开浏览器的可执行文件：`dist/bucad-demo/bucad-demo.exe`。
- 桌面窗口版可执行文件：`dist/bucad-demo-desktop/bucad-demo-desktop.exe`。
- 发布或拷贝桌面版 demo 时需要带上整个生成目录，不能只单独拷贝 `.exe` 文件，因为程序依赖同目录下的模型文件、Python 库、WebView 文件和配置文件。
- `v1.1.0` release 包使用桌面窗口版，启动后界面表现为本地 Windows 应用，而不是打开外部浏览器。

## 批量评估

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_demo_convnext_effnet.json
```

打包 demo 配置的代表性指标：

| 模型 | 阈值 | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate | 0.510 | 0.9256 | 0.8532 | 0.8667 | 0.7309 | 0.8467 | 0.7930 |

## 生成数据划分

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

输出：

- `artifacts/reports/busbra_5fold_splits.csv`
- `artifacts/reports/busbra_split_summary.json`

## 训练分类模型

训练 ConvNeXt-Tiny 单折：

```powershell
python scripts\train_cls.py --config configs\classifier\convnext_tiny_timm_recipe.yml --fold 1
```

训练 EfficientNetV2-S 单折：

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

五折 checkpoint 输出到 `artifacts/checkpoints/`，不提交到远程仓库。

## 实验报告与结果文件

如需进一步了解模型选择、消融实验和历史对比结果，可以查看以下报告文件：

- `artifacts/reports/fivefold_single_model_comparison.md`
- `artifacts/reports/fold1_single_model_baseline_comparison.md`
- `artifacts/reports/convnext_tta_optimization.md`
- `artifacts/reports/ensemble_tta_threshold_tuning.md`
- `artifacts/reports/roi_oof_experiment.md`
- `artifacts/reports/roi_oof_lcc_optimization.md`
- `artifacts/reports/roi_precision_f1_study.md`
- `artifacts/reports/oof_two_model_decision_report.md`
- `artifacts/reports/four_model_ensemble_weight_search.md`
- `artifacts/reports/three_model_ensemble_weight_search.md`
- `artifacts/reports/three_model_roi_oof_experiment.md`
- `artifacts/reports/swin_tiny_5fold_experiment.md`
- `artifacts/reports/报告/中文版报告/training_recipe_audit.md`
- `artifacts/reports/报告/中文版报告/literature_guided_optimization.md`
- `artifacts/reports/报告/中文版报告/final_validation.md`

英文版历史报告统一存放在 `artifacts/reports/报告/英文版报告/`，对应中文版同步存放在 `artifacts/reports/报告/中文版报告/`。

## 参考与致谢

本项目在设计和优化过程中参考了公开乳腺超声数据集、医学影像开源项目，以及分类、分割、ROI 感知诊断、多任务学习和超声基础模型相关研究。

- BUSI 数据集：[Dataset of breast ultrasound images](https://pubmed.ncbi.nlm.nih.gov/31867417/)
- 乳腺超声病灶区域感知分类研究：[PMC11431713](https://pmc.ncbi.nlm.nih.gov/articles/PMC11431713/)
- 乳腺超声分割与分类多任务研究：[PMC12011763](https://pmc.ncbi.nlm.nih.gov/articles/PMC12011763/)
- OpenUS 超声基础模型：[XZheng0427/OpenUS](https://github.com/XZheng0427/OpenUS)
- BUSI 分割参考项目：[tqxli/breast_ultrasound_lesion_segmentation_PyTorch](https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch)
- BUSI-SAM / SAM 风格分割参考：[huangjin520/BUSI-SAM](https://github.com/huangjin520/BUSI-SAM)、[bscs12/BUSSAM](https://github.com/bscs12/BUSSAM)

感谢上述数据集、论文和开源项目的作者与维护者。他们的工作为 BUCAD 的数据处理、模型对比、分割可视化、混合集成设计和后续 ROI 感知优化提供了重要参考。
