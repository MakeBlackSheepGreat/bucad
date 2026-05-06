# BUCAD - 乳腺超声计算机辅助诊断系统

BUCAD（Breast Ultrasound Computer-Aided Diagnosis）是一个面向乳腺超声影像的计算机辅助诊断研究原型。系统整合了深度学习分类模型、语义分割模型、ROI 引导推理、Grad-CAM 可解释性可视化、Gradio 交互界面及 Windows 桌面部署方案，构成一个完整的算法验证与原型演示框架。

本系统适用于医学影像算法研究、模型对比实验、教学演示及二次开发。当前版本已在 BUSBRA 训练集和 BUSI 外部测试集上完成验证。

> **声明**：BUCAD 为科研原型系统，非临床诊断产品，不可替代专业医学判断。

## 系统功能

输入单张乳腺超声图像后，系统输出包括：

- 良性/恶性二分类概率及基于可配置阈值的判定结果
- 预测置信度与边界样本标记
- 基于语义分割的病灶区域定位叠加图
- 基于 Grad-CAM 的分类模型注意力热力图
- 批量数据集评估与可复现实验报告

### 推理流程

```
输入图像 → 灰度转换 + CLAHE 增强 + 尺寸缩放 + 模型专属归一化
    ↓
五折分类模型集成 → 完整图良恶性概率
    ↓
语义分割模型 → 病灶 mask → 最大连通域提取 + 边界扩展裁剪 → ROI 图像
    ↓
同一分类模型集成 → ROI 良恶性概率
    ↓
OOF 训练的 Logistic Stacking → 融合完整图与 ROI 概率
    ↓
阈值判定 + 置信度计算 + 边界标记
    ↓
输出：分类结果 + 分割叠加图 + Grad-CAM 热力图
```

## 模型架构

### 当前主线配置

当前部署配置位于 `configs/inference/demo.yml`，采用 `ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate` 双模型集成架构。

| 组件 | 模型 | 权重 | Checkpoint 数量 | TTA 策略 |
|---|---|---|---|---|
| 主分类分支 | ConvNeXt-Tiny | 0.573 | 5-fold | crop-sweep (0.90/0.95/1.00) + hflip |
| 辅助分类分支 | EfficientNetV2-S | 0.427 | 5-fold | identity |
| 分割分支 | UNet (ResNet-18 encoder) | — | 1 | — |
| 融合层 | Logistic Regression Stacker | — | — | — |

### 关键技术参数

| 参数 | 值 | 说明 |
|---|---|---|
| 分割 mask 阈值 | 0.40 | 病灶概率图二值化阈值 |
| ROI 边界扩展系数 | 0.35 | mask bbox 外扩比例 |
| ROI 面积质量门控 | [0.08, 0.75] | 超出范围回退至完整图预测 |
| 分类阈值 | 0.510 | 基于 OOF 证据确定的运行点 |
| 边界样本标记阈值 | 0.08 | 概率距阈值 ±0.08 内标记为不确定 |

### 代表性评估结果

| 数据集 | 阈值 | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| BUSBRA (5-fold CV) | 0.510 | 0.9256 | 0.8532 | 0.8667 | 0.8467 | 0.7309 | 0.7930 |
| BUSI (外部测试) | 0.540 | 0.8991 | 0.8346 | 0.7381 | 0.8810 | 0.7488 | 0.7434 |

## 模型筛选与对比

### 单模型基线对比

项目采用统一流程对多个候选模型进行筛选，以下为单折（fold1）原生训练配方的 BUSI 外部测试结果：

| 模型 | 参数量 | AUC | Accuracy | Sensitivity | Specificity | F1-Score |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny (timm recipe) | 28M | 0.8943 | 0.8423 | 0.7762 | 0.8741 | 0.7617 |
| ConvNeXt-Small (timm recipe) | 50M | 0.8947 | 0.8284 | 0.7667 | 0.8581 | 0.7436 |
| DenseNet-121 | 8M | 0.8766 | 0.8083 | 0.4571 | 0.9771 | 0.6076 |
| Swin-Tiny (timm recipe) | 28M | 0.8729 | 0.8300 | 0.7048 | 0.8902 | 0.7291 |
| EfficientNetV2-S | 21M | 0.8609 | 0.7465 | 0.8333 | 0.7048 | 0.6809 |
| ResNet-18 | 11M | 0.8480 | 0.7991 | 0.7714 | 0.8124 | 0.7137 |
| MobileNetV3-Small | 2.5M | 0.8431 | 0.7543 | 0.7143 | 0.7735 | 0.6536 |

### 五折集成对比

六个模型家族的 5-fold 交叉验证及 BUSI 外部测试结果：

| 模型 | BUSBRA 5-fold AUC (均值±标准差) | BUSI AUC | BUSI Sensitivity | BUSI Specificity | Youden 阈值 |
|---|---|---:|---:|---:|---:|
| ConvNeXt-Tiny V1 | 0.9212 ± 0.0196 | 0.8991 | 0.7381 | 0.8810 | 0.54 |
| ConvNeXt-Tiny V2 | 0.9176 ± 0.0107 | 0.8897 | 0.7333 | 0.8970 | 0.39 |
| ConvNeXt-Small | 0.9162 ± 0.0163 | 0.9040 | 0.8190 | 0.8535 | 0.47 |
| Swin-Tiny | 0.9139 ± 0.0130 | 0.8831 | 0.7476 | 0.8696 | 0.61 |
| DenseNet-121 | 0.9071 ± 0.0174 | 0.8909 | 0.3905 | 0.9908 | 0.18 |
| EfficientNetV2-S | 0.8946 ± 0.0241 | 0.8982 | 0.6619 | 0.9314 | 0.39 |

> 详细对比数据见 `artifacts/reports/six_model_comparison_report.md` 和 `artifacts/reports/six_model_comparison_data.json`

### 可选集成配置

| 配置 | 模型组合 | 阈值 | AUC | Sensitivity | Specificity | F1-Score |
|---|---|---:|---:|---:|---:|---:|
| 三模型非 ROI | ConvNeXt + EfficientNet + DenseNet | 0.453 | 0.9162 | 0.7476 | 0.9291 | 0.7889 |
| 三模型 ROI OOF | ConvNeXt + EfficientNet + DenseNet + ROI | 0.560 | 0.9229 | 0.8143 | 0.8673 | 0.7790 |

## 技术设计依据

### 五折交叉验证集成

受限于训练数据规模，单一 train/val 划分难以充分覆盖病灶形态、BI-RADS 分级、采集条件及良恶性边界样本的分布多样性。五折集成通过 StratifiedGroupKFold 按病例级分组划分，确保同一患者的样本不跨折出现，避免数据泄漏。推理时对五个折的输出取加权平均，降低单折偶然性，提升概率输出稳定性。

### 主模型选择：ConvNeXt-Tiny

ConvNeXt-Tiny 在采用 timm-aware 训练配方（ImageNet 预训练均值/标准差、bicubic 插值、timm crop 配置、类别平衡权重、best-AUC 检查点选择）后，表现出最优的恶性样本排序能力。早期非 timm-aware 配方下 ConvNeXt 性能显著下降（AUC 从 0.93 降至 0.60），表明该模型对预处理策略高度敏感。

在当前双模型集成中，ConvNeXt-Tiny 承担主要的恶性检出任务（权重 0.573），其较高的 Sensitivity 降低了漏诊风险。

### 辅助模型选择：EfficientNetV2-S

EfficientNetV2-S 单独部署时 Sensitivity 偏低（0.66），但 Specificity 较高（0.93），属于保守型分类器。将其纳入集成可与 ConvNeXt-Tiny 形成互补：ConvNeXt 偏向恶性召回，EfficientNet 偏向良性特异，两者错误分布的差异性提升了集成的整体鲁棒性。

### Crop-Sweep 测试时增强

乳腺超声图像中病灶尺寸、位置及周围组织背景差异显著。单一 center crop 可能因裁剪过紧丢失病灶周围组织信息，或因裁剪过松引入过多无关背景。

ConvNeXt 分支采用三种 crop 比例（0.90、0.95、1.00），每种配合水平翻转，共生成六个推理视图。多视图预测结果取平均，降低了单一裁剪策略失败的风险。EfficientNetV2-S 分支因独立测试 TTA 收益有限且增加推理延迟，仅保留 identity TTA。

### CLAHE 预处理与模型专属归一化

超声图像普遍存在局部对比度低、散斑噪声强、病灶边缘模糊等问题。CLAHE（Contrast Limited Adaptive Histogram Equalization）可增强局部对比度，有利于 CNN 特征提取器捕捉病灶边界、内部回声及周围组织差异。

不同模型因训练时接触的数据分布不同，推理时必须保持预处理一致性。ConvNeXt-Tiny 使用 ImageNet 标准化参数和 bicubic 插值；EfficientNetV2-S 使用 area 插值且不继承 ConvNeXt 的归一化参数。跨模型预处理不匹配会导致性能退化。

### ROI 分割引导

完整图分类器接收整张超声图像，包含黑边、设备标注、探头区域及正常组织纹理等无关信息，这些区域可能干扰分类决策。

ROI 分支通过语义分割模型预测病灶 mask，经最大连通域提取和边界扩展裁剪后生成 ROI 图像，再由同一分类模型评估局部病灶视角下的恶性概率。ROI 分支与完整图分支并行运作：完整图保留全局上下文，ROI 图强调病灶本体形态，两者通过 OOF 训练的 Logistic Stacker 融合。

### ROI 面积质量门控

分割预测并非始终可靠。面积过小（< 0.08）可能表示分割器仅捕获噪声区域或遗漏真实病灶；面积过大（> 0.75）则 mask 几乎覆盖全图，ROI 失去聚焦意义。面积门控机制在 ROI 质量异常时回退至完整图预测，作为分割失败的安全兜底策略。

### OOF Logistic Stacking

完整图与 ROI 图的概率分布存在差异，直接平均会引入校准问题。系统采用 BUSBRA 训练集 out-of-fold 预测训练的 Logistic Regression 作为融合层，以 logit 空间特征作为输入，学习完整图为主信号、ROI 为辅助校正信号的最优权重分配。

### 阈值选择

默认阈值 0.510 基于训练集 OOF 证据确定，并在 BUSI 外部测试中验证（Youden 最优阈值同样为 0.51）。医学筛查任务中，漏诊恶性病例的代价通常高于良性误报，因此阈值选择优先保证 Sensitivity，同时控制 Specificity 不显著下降。

## 指标定义

| 指标 | 定义 | 临床意义 |
|---|---|---|
| AUC | ROC 曲线下面积 | 模型区分良恶性的整体排序能力 |
| Accuracy | (TP+TN) / (TP+TN+FP+FN) | 整体预测正确率 |
| Sensitivity (Recall) | TP / (TP+FN) | 恶性检出率，反映漏诊风险 |
| Specificity | TN / (TN+FP) | 良性正确识别率，反映误诊控制 |
| Precision | TP / (TP+FP) | 阳性预测可靠性 |
| F1-Score | 2 × Precision × Recall / (Precision + Recall) | Precision 与 Recall 的调和均值 |
| Youden's J | Sensitivity + Specificity - 1 | 综合评估阈值优劣 |

## 项目结构

```
BUCAD/
├── configs/                          # 配置文件
│   ├── classifier/                   # 分类器训练配置
│   ├── segmenter/                    # 分割器训练配置
│   └── inference/                    # 推理与集成配置
├── src/
│   ├── datasets/                     # 数据集加载与划分
│   ├── models/                       # 模型工厂（分类器/分割器）
│   ├── engine/                       # 训练/评估/推理引擎
│   ├── explain/                      # Grad-CAM 可解释性
│   ├── preprocess/                   # 图像预处理与 ROI 裁剪
│   └── utils/                        # 配置/指标/报告/日志
├── scripts/                          # 命令行入口脚本
├── app/                              # Gradio Web 应用
├── packaging/                        # Windows 桌面打包配置
├── tests/                            # 单元/集成/smoke 测试
└── artifacts/
    ├── checkpoints/                  # 模型权重（不提交至 Git）
    └── reports/                      # 实验报告与评估结果
```

## 环境配置

### 依赖安装

```powershell
conda create -n BUCAD python=3.11 -y
conda activate BUCAD
python -m pip install -r requirements.txt
```

### 环境验证

```powershell
python check_env.py
python check_all.py
```

### 数据集配置

复制 `configs/paths.example.yml` 为 `configs/paths.local.yml`，配置本地数据集路径：

```yaml
datasets:
  busbra_root: ./BUSBRA
  busi_root: ./Dataset_BUSI_with_GT
```

数据集采用病例级划分（StratifiedGroupKFold），避免同一患者样本跨训练/验证集出现导致数据泄漏。

## 运行方式

### Web 界面

```powershell
conda activate BUCAD
python app\main.py
```

### 桌面窗口模式

```powershell
conda activate BUCAD
python app\desktop_main.py
```

两种模式均读取 `configs/inference/demo.yml`。

### Windows 桌面打包

```powershell
# 浏览器模式
python -m PyInstaller --clean --noconfirm packaging\demo.spec

# 桌面窗口模式
python -m PyInstaller --clean --noconfirm packaging\desktop_demo.spec
```

输出路径：
- 浏览器模式：`dist/bucad-demo/bucad-demo.exe`
- 桌面窗口模式：`dist/bucad-demo-desktop/bucad-demo-desktop.exe`

> 桌面版需携带完整生成目录，不可单独分发 `.exe` 文件。

### 数据划分生成

```powershell
python scripts\make_split.py --config configs\paths.local.yml
```

输出：
- `artifacts/reports/busbra_5fold_splits.csv`
- `artifacts/reports/busbra_split_summary.json`

### 分类模型训练

```powershell
# ConvNeXt-Tiny fold 1
python scripts\train_cls.py --config configs\classifier\convnext_tiny_timm_recipe.yml --fold 1

# EfficientNetV2-S fold 1
python scripts\train_cls.py --config configs\classifier\efficientnetv2_s.yml --fold 1
```

五折 checkpoint 输出至 `artifacts/checkpoints/`。

### 批量评估

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_demo.json
```

## 硬件要求

### 推理/演示

| 项目 | 最低要求 | 推荐配置 |
|---|---|---|
| 操作系统 | Windows 10 x64 | Windows 11 x64 |
| CPU | 支持 AVX2 指令集 | 现代多核处理器 |
| GPU | 非必需（支持 CPU 推理） | NVIDIA GPU（降低推理延迟） |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 8 GB | — |

### 训练/实验

| 项目 | 最低要求 | 推荐配置 |
|---|---|---|
| Python | 3.10 / 3.11 (Conda) | 3.11 |
| GPU | NVIDIA CUDA，8 GB VRAM | 12-16 GB VRAM |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 50 GB | — |

## 实验报告索引

| 报告文件 | 内容 |
|---|---|
| `artifacts/reports/six_model_comparison_report.md` | 六模型全面对比（BUSBRA + BUSI） |
| `artifacts/reports/six_model_comparison_data.json` | 对比数据（JSON 格式，供程序读取） |
| `artifacts/reports/fivefold_single_model_comparison.md` | 五折单模型对比 |
| `artifacts/reports/fold1_single_model_baseline_comparison.md` | Fold1 单模型基线对比 |
| `artifacts/reports/native_single_model_retest.md` | 原生配方单模型重测 |
| `artifacts/reports/convnext_tta_optimization.md` | ConvNeXt TTA 优化实验 |
| `artifacts/reports/ensemble_tta_threshold_tuning.md` | 集成 TTA 与阈值调优 |
| `artifacts/reports/roi_oof_experiment.md` | ROI OOF 融合实验 |
| `artifacts/reports/roi_oof_lcc_optimization.md` | ROI OOF + 最大连通域优化 |
| `artifacts/reports/roi_precision_f1_study.md` | ROI 对 Precision/F1 的影响研究 |
| `artifacts/reports/oof_two_model_decision_report.md` | 双模型 OOF 决策报告 |
| `artifacts/reports/four_model_ensemble_weight_search.md` | 四模型集成权重搜索 |
| `artifacts/reports/three_model_ensemble_weight_search.md` | 三模型集成权重搜索 |
| `artifacts/reports/three_model_roi_oof_experiment.md` | 三模型 ROI OOF 实验 |
| `artifacts/reports/swin_tiny_5fold_experiment.md` | Swin-Tiny 五折实验 |

## 参考文献

- Al-Dhabyani W, et al. Dataset of breast ultrasound images. *Data in Brief*, 2020. [[PubMed]](https://pubmed.ncbi.nlm.nih.gov/31867417/)
- ROI-aware classification for breast ultrasound. [[PMC11431713]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11431713/)
- Multi-task learning for breast ultrasound segmentation and classification. [[PMC12011763]](https://pmc.ncbi.nlm.nih.gov/articles/PMC12011763/)
- OpenUS: Ultrasound foundation model. [[GitHub]](https://github.com/XZheng0427/OpenUS)
- BUSI segmentation reference. [[GitHub]](https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch)
- BUSI-SAM / BUSSAM segmentation references. [[GitHub]](https://github.com/huangjin520/BUSI-SAM) [[GitHub]](https://github.com/bscs12/BUSSAM)

## 许可

本项目仅供科研与教学使用。
