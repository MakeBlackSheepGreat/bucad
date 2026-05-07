# BUCAD

[English README](README.md)

BUCAD（Breast Ultrasound Computer-Aided Diagnosis）是一个面向乳腺超声影像分析的 Windows 优先科研原型系统。工程整合了良恶性分类、病灶区域引导、可解释性可视化、本地 Gradio 推理界面以及 Windows 桌面打包流程。

该项目用于算法验证、可复现实验、教学演示和受控的二次开发，不是临床诊断产品，不能替代医生判断。

## 适用范围与验证边界

- BUSBRA 用于训练、内部验证、out-of-fold 选择和候选方案筛选。
- BUSI 仅用于候选方案冻结之后的锁定外部验证与阈值确认。
- 项目中的对比表统一报告 AUC、Accuracy、Recall/Sensitivity、Precision、Specificity 和 F1-Score。

## 已冻结主线

当前部署配置位于 `configs/inference/demo.yml`。

- 主分类分支：`ConvNeXt-Tiny`，五折 checkpoint，集成权重 `0.573`，使用 timm-aware 预处理和 crop-sweep TTA。
- 辅助分类分支：`EfficientNetV2-S`，五折 checkpoint，集成权重 `0.427`，使用 CLAHE 预处理和 identity TTA。
- ROI 分支：`segmenter_fold1.pt` 用于生成病灶 mask；训练集分割验证确定 `mask_threshold=0.40`、`margin_ratio=0.35` 和最大连通域裁剪。
- 融合策略：完整图概率与 ROI 概率在 logit 空间中通过逻辑回归式 stacker 融合，stacker 由 BUSBRA 的 out-of-fold 预测训练得到。
- 质量门控：当 ROI 面积比例低于 `0.08` 或高于 `0.75` 时，回退到完整图分支。
- 默认运行阈值：`0.510`。

锁定后的 BUSI 外部验证结果：

| 模型 | 阈值 | AUC | Accuracy | Recall/Sensitivity | Precision | Specificity | F1-Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF Stacking + LCC ROI + Area Gate | 0.510 | 0.9256 | 0.8532 | 0.8667 | 0.7309 | 0.8467 | 0.7930 |

混淆矩阵：TN 370 / FP 67 / FN 28 / TP 182。

## 代表性单模型筛选

完整筛选记录见 [`artifacts/reports/native_single_model_retest.md`](artifacts/reports/native_single_model_retest.md)。

| 模型 | AUC | Recall/Sensitivity | Specificity | 说明 |
| --- | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny timm recipe fold1 | 0.8943 | 0.7762 | 0.8741 | 推荐 ConvNeXt 配方 |
| ConvNeXt-Small timm recipe fold1 | 0.8947 | 0.7667 | 0.8581 | 后续升级参考 |
| Swin-Tiny timm recipe fold1 | 0.8729 | 0.7048 | 0.8902 | Transformer 风格候选 |
| DenseNet121 fold1 | 0.8766 | 0.4571 | 0.9771 | 高特异性参考 |
| EfficientNetV2-S fold1 | 0.8609 | 0.8333 | 0.7048 | 高灵敏度 CNN 候选 |
| ResNet18 fold1 | 0.8480 | 0.7714 | 0.8124 | 基线参考 |
| MobileNetV3-Small fold1 | 0.8431 | 0.7143 | 0.7735 | 轻量基线 |
| Basic CNN fold1 | 0.7327 | 0.0429 | 0.9794 | 非预训练基线 |
| VGG16 fold1 | 0.5000 | 0.0000 | 1.0000 | 基线失败案例 |

早期非 timm-aware 的 ConvNeXt-Tiny 结果 AUC 仅为 0.5996，因此没有进入冻结主线。引入 timm-aware 配方后，该分支才具备稳定的排序能力并成为主模型。

## 设计依据

- 五折集成用于降低小样本条件下的方差，并减轻单次划分对结果的偶然影响。
- ConvNeXt-Tiny 作为主分支，是因为 timm-aware 配方修正了训练与推理预处理不一致的问题，并获得了更稳定的排序表现。
- EfficientNetV2-S 作为辅助分支，是因为它的误差分布与 ConvNeXt-Tiny 互补，且更偏向特异性稳定。
- 异构预处理被保留，是因为不同模型家族对归一化、插值和裁剪策略的敏感性不同。
- ROI 引导用于缓解黑边、文本标注、探头伪影、背景组织和正常纹理对分类器的干扰。
- 最大连通域和非零边界扩展用于抑制碎片化 mask，同时保留边界上下文。
- 完整图与 ROI 概率在 logit 空间融合，是因为两者的校准特性并不一致。
- ROI 面积门控用于避免不稳定病灶裁剪对最终运行点造成过大影响。
- `0.510` 阈值来自 BUSBRA 的 out-of-fold 证据，并在锁定 BUSI 外部验证中确认。
- Grad-CAM 与叠加图输出围绕 ConvNeXt-Tiny 对齐，以保持解释对象稳定。

## 已拒绝实验

已测试但未并入 `configs/inference/demo.yml` 的方案包括：五折分割器、model-zoo stacking、hard-sample weighting、seed diversity、weight soup、soft ROI gate、EfficientNet TTA、CutMix、轻量正则化和 320 输入重训练。

其中一部分在 BUSBRA OOF 上有提升，但未能迁移到 BUSI，或者部署复杂度增加而外部结果没有改善。这些方案保留在 `artifacts/reports/` 中，不进入冻结主线。

## 目录结构

- `configs/`：路径、分类器、分割器与推理配置。
- `src/datasets/`：数据加载与划分支持。
- `src/models/`：分类器与分割器工厂。
- `src/engine/`：训练、对比、推理与评估流程。
- `src/explain/`：Grad-CAM 与叠加图生成。
- `src/preprocess/`：图像读写与预处理变换。
- `src/utils/`：配置、指标、报告、路径、日志和结果结构。
- `scripts/`：命令行入口。
- `app/`：Gradio 网页应用。
- `tests/`：单元、集成和 smoke 测试。
- `artifacts/reports/`：带版本的实验总结与基准说明。
- `artifacts/checkpoints/`：本地模型权重，Git 忽略。

## 数据路径

先复制 `configs/paths.example.yml` 为 `configs/paths.local.yml`，再指向本地数据集：

```yaml
datasets:
  busbra_root: ./BUSBRA
  busi_root: ./Dataset_BUSI_with_GT
```

说明：

- 数据集目录可按本机实际结构调整名称。
- split 采用病例级划分，避免数据泄漏。
- 数据集、checkpoint、生成图片、JSON/CSV 输出和发布包均为本地产物，不进入 Git。

## 环境安装

```powershell
conda create -n BUCAD python=3.11 -y
conda activate BUCAD
python -m pip install -r requirements.txt
```

环境检查：

```powershell
python check_env.py
python check_all.py
```

## 硬件与软件要求

### demo / 推理

- **操作系统**：Windows 10/11 x64 是主要目标。
- **打包桌面版**：解压发布包后不需要本机额外安装 Python；桌面版依赖 Microsoft Edge WebView2。
- **CPU / GPU**：支持 CPU 推理，NVIDIA GPU 主要用于缩短响应时间。
- **内存**：最低 8 GB，建议 16 GB。
- **磁盘**：建议至少预留 8 GB，用于解压后的 demo、权重、临时文件和可视化结果。

### 训练 / 实验

- **Python 环境**：Conda + Python 3.10 或 3.11。
- **GPU**：完整五折训练建议使用 NVIDIA CUDA GPU。
- **显存**：当前 224 分辨率的 ConvNeXt-Tiny / EfficientNetV2-S 实验，8 GB 为实用下限，12-16 GB 更合适。
- **系统内存**：最低 16 GB，建议 32 GB。
- **磁盘**：建议至少预留 50 GB，用于数据集、checkpoint、日志、OOF 产物、报告和临时构建文件。
- **数据管理**：数据集、checkpoint、生成报告和发布包应保留在 Git 追踪源码之外。

## 运行 Demo

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

两种模式均读取 `configs/inference/demo.yml`。当前 demo 以 ConvNeXt-Tiny 为主分支，以 EfficientNetV2-S 为辅助分支。

## Windows 一键打包

浏览器打开版：

```powershell
conda activate BUCAD
python -m PyInstaller --clean --noconfirm packaging\demo.spec
```

桌面窗口版：

```powershell
conda activate BUCAD
python -m PyInstaller --clean --noconfirm packaging\desktop_demo.spec
```

- 浏览器打开版可执行文件：`dist/bucad-demo/bucad-demo.exe`
- 桌面窗口版可执行文件：`dist/bucad-demo-desktop/bucad-demo-desktop.exe`
- 需要分发完整生成目录，不建议只拷贝单个 `.exe`，因为程序依赖同目录下的权重、Python 库、WebView 文件和配置。
- `v1.1.0` 发布包使用桌面窗口版，以本地 Windows 应用的形式启动界面。

## 批量评估

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_demo_convnext_effnet.json
```

## 生成划分

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

输出：

- `artifacts/reports/busbra_5fold_splits.csv`
- `artifacts/reports/busbra_split_summary.json`

## 训练分类器

训练一个 fold：

```powershell
python scripts\train_cls.py --config configs\classifier\convnext_tiny_timm_recipe.yml --fold 1
```

训练 EfficientNetV2-S 一个 fold：

```powershell
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

五折 checkpoint 生成到 `artifacts/checkpoints/`，默认不提交到 Git。

## 关键报告

- [`artifacts/reports/native_single_model_retest.md`](artifacts/reports/native_single_model_retest.md)
- [`artifacts/reports/optimization_attempts_2026_04_27.md`](artifacts/reports/optimization_attempts_2026_04_27.md)
- [`artifacts/reports/threshold_analysis.md`](artifacts/reports/threshold_analysis.md)
- [`artifacts/reports/competition_metrics_report_regression.md`](artifacts/reports/competition_metrics_report_regression.md)
- [`artifacts/reports/competition_metrics_audit.md`](artifacts/reports/competition_metrics_audit.md)

## 参考文献

本项目在设计和优化过程中参考了公开乳腺超声数据集、医学影像开源项目，以及分类、分割、ROI 感知诊断、多任务学习和超声基础模型相关研究。

- BUSI 数据集：[Dataset of breast ultrasound images](https://pubmed.ncbi.nlm.nih.gov/31867417/)
- 病灶区域感知乳腺超声分类：[PMC11431713](https://pmc.ncbi.nlm.nih.gov/articles/PMC11431713/)
- 乳腺超声分割与分类多任务研究：[PMC12011763](https://pmc.ncbi.nlm.nih.gov/articles/PMC12011763/)
- OpenUS 超声基础模型：[XZheng0427/OpenUS](https://github.com/XZheng0427/OpenUS)
- BUSI 分割参考项目：[tqxli/breast_ultrasound_lesion_segmentation_PyTorch](https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch)
- BUSI-SAM / SAM 风格分割参考：[huangjin520/BUSI-SAM](https://github.com/huangjin520/BUSI-SAM)，[bscs12/BUSSAM](https://github.com/bscs12/BUSSAM)

这些参考为 BUCAD 的数据处理、模型对比、分割可视化、集成设计和 ROI 感知优化提供了依据。
