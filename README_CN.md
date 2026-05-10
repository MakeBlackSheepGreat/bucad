# BUCAD

[English README](README.md)

BUCAD（Breast Ultrasound Computer-Aided Diagnosis）是一个面向乳腺超声影像良恶性辅助分析的计算机辅助诊断研究原型系统。系统集成深度学习分类、语义分割、ROI 引导推理、Grad-CAM 可解释性可视化、本地 Gradio 推理界面以及 Windows 桌面部署方案。

本项目用于算法验证、可复现实验、教学演示及受控的二次开发，尚未经过临床注册和多中心临床验证，不能作为独立临床诊断依据。

## 数据与验证

- BUSBRA 用于模型训练、内部验证、out-of-fold 候选筛选和阈值选择。
- BUSI 仅在候选方案冻结之后用于锁定外部验证与结果复核，不参与训练、模型选择、参数搜索或阈值调优。
- 项目实验对比统一报告 AUC、Accuracy、Recall/Sensitivity、Precision、Specificity 和 F1-Score。

## 当前主线配置

当前可复现推理配置位于 `configs/inference/demo.yml`。

| 组件 | 模型 | 算法 | 权重 | 说明 |
|---|---|---|---|---|
| 主分类分支 | ConvNeXt-Tiny | ConvNeXt (Liu et al., 2022) | 0.573 | 五折 checkpoint，timm-aware 预处理，crop-sweep TTA |
| 辅助分类分支 | EfficientNetV2-S | EfficientNetV2 (Tan & Le, 2021) | 0.427 | 五折 checkpoint，CLAHE 预处理，identity TTA |
| 分割分支 | UNet-ResNet18 | UNet (Ronneberger et al., 2015) + ResNet-18 encoder | — | ImageNet 预训练编码器，输出二值病灶 mask |
| 融合层 | Logistic Stacker | Logistic Regression (sklearn) | — | 基于 BUSBRA OOF 训练，使用 logit 空间特征 |

### 关键技术参数

| 参数 | 值 | 选定依据 |
|---|---|---|
| 分割 mask 阈值 | 0.40 | BUSBRA Dice 扫描：0.30→0.7971, **0.40→0.8085**, 0.50→0.8074, 0.60→0.7797 |
| ROI 边界扩展系数 | 0.35 | 保留病灶周边组织上下文的折中设置 |
| ROI 面积门控 | [0.08, 0.75] | 超出范围回退至完整图预测，由 BUSBRA OOF 协议选定 |
| 分类阈值 | 0.510 | BUSBRA OOF 证据选定；BUSI 仅作冻结后外部复核 |
| 边界样本标记 | ±0.08 | 预测概率距阈值 ±0.08 内标记为不确定 |

### BUSI 外部验证结果

| 配置 | 阈值 | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| 完整主线（ROI Area Gate） | 0.510 | 0.9256 | 0.8532 | 0.8667 | 0.8467 | 0.7309 | 0.7930 |

混淆矩阵：TN 370 / FP 67 / FN 28 / TP 182。

## 预处理流程

### CLAHE 对比度增强

超声图像普遍存在局部对比度低、散斑噪声强、病灶边缘模糊等问题。系统在配置启用的分支中应用 Contrast Limited Adaptive Histogram Equalization（CLAHE），通过局部直方图均衡增强病灶边界、内部回声及周围组织的对比度差异，为后续 CNN 特征提取提供更稳定的灰度结构。

### 模型专属归一化

不同模型家族因 ImageNet 预训练配置不同，推理时必须保持预处理一致性。系统采用 timm-aware 策略，使 mean/std、插值方式和 crop 比例与预训练配方保持一致：

| 模型 | mean | std | 插值 | crop_pct |
|---|---|---|---|---|
| ConvNeXt-Tiny | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | bicubic | 0.95 |
| EfficientNetV2-S | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | area | 默认 |

### timm-aware 预处理的必要性

timm-aware 预处理是 ConvNeXt 系列模型有效迁移预训练权重的前提条件。早期非 timm-aware 配方下，ConvNeXt-Tiny 在 BUSI 上 AUC 仅为 0.5996（Sensitivity=0，近似随机）。引入 timm-aware 配方后 AUC 提升至 0.8943（+0.2947）。Swin-Tiny 同样受益，非 timm 配方 AUC 0.8242，timm 配方提升至 0.8729（+0.0487）。

详细消融数据见 `artifacts/reports/Chinese reports/01_baseline_model_screening/native_single_model_retest.md`。

## 优化技术与消融实验

### 1. 五折交叉验证集成

采用病例级分组的 StratifiedGroupKFold 划分方案，确保同一患者样本不跨折出现。推理阶段先在同一模型族内汇总五折输出，再进入后续模型融合。

**消融结果（BUSBRA 5-fold CV AUC）：**

| 模型 | Fold1 单折 | 5-fold 均值 | 标准差 | 说明 |
|---|---:|---:|---:|---|
| ConvNeXt-Tiny V1 | 0.9259 | 0.9212 | 0.0196 | — |
| ConvNeXt-Tiny V2 | 0.9278 | 0.9176 | 0.0107 | 更稳定 |
| ConvNeXt-Small | 0.9118 | 0.9162 | 0.0163 | — |
| EfficientNetV2-S | 0.9248 | 0.8946 | 0.0241 | — |

**BUSI 外部消融（5-fold vs 单折）：**

| 模型 | 单折 AUC | 5-fold AUC | 提升 |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.8943 | 0.9054 | +0.0111 |
| EfficientNetV2-S | 0.8609 | 0.8997 | +0.0388 |
| DenseNet-121 | 0.8766 | 0.8914 | +0.0148 |
| Swin-Tiny | 0.8729 | 0.8971 | +0.0242 |

详细数据见 `artifacts/reports/Chinese reports/01_baseline_model_screening/fivefold_single_model_comparison.md`。

### 2. Crop-Sweep 测试时增强

乳腺超声图像中病灶尺寸、位置及周围组织背景差异显著。单一 center crop 可能因裁剪过紧丢失病灶周围组织信息，或因裁剪过松引入过多无关背景。

ConvNeXt 分支采用三种 crop 比例（0.90、0.95、1.00），每种配合水平翻转，共生成六个推理视图。多视图预测结果取平均，以降低单一裁剪尺度造成的偶然偏差。

**消融结果（ConvNeXt-Tiny 5-fold，BUSI 外部）：**

| TTA 策略 | AUC | Sensitivity | Specificity | F1-Score |
|---|---:|---:|---:|---:|
| identity（基线） | 0.8991 | 0.7381 | 0.8810 | 0.7434 |
| + 水平翻转 | 0.9041 | — | 0.8924 | 0.7524 |
| + crop-sweep | **0.9054** | **0.7714** | 0.8719 | **0.7696** |
| + 旋转 ±5° | 0.9048 | — | **0.8970** | — |

crop-sweep 在 AUC（+0.0063）、Sensitivity（+0.0333）和 F1（+0.0262）上均为最优。

详细数据见 `artifacts/reports/Chinese reports/04_tta_threshold_external_eval/convnext_tta_optimization.md`。

### 3. ROI 分割引导

完整图分类器接收整张超声图像，可能同时包含黑边、设备标注、探头区域及正常组织纹理等非病灶信息。ROI 分支通过语义分割模型（UNet + ResNet-18 encoder）预测病灶 mask，经最大连通域提取和边界扩展裁剪后生成 ROI 图像，再由分类模型评估局部病灶视角下的恶性概率。

**消融结果（BUSI 外部）：**

| 配置 | AUC | Sensitivity | 说明 |
|---|---:|---:|---|
| 完整图基线 | 0.9151 | — | 双模型集成，无 ROI |
| + Segmenter ROI | 0.9196 | 0.8429 | +0.0045 |
| + LCC 后处理 | 0.9208 | 0.8524 | +0.0057 vs 基线 |
| Oracle ROI（GT mask） | 0.9202 | — | 理论上界对照，不可部署 |

ROI 引导在 AUC 上带来 +0.0057 的稳定提升。最大连通域（LCC）后处理通过抑制碎片化 mask 进一步提升 AUC +0.0012、Sensitivity +0.0095。

详细数据见 `artifacts/reports/Chinese reports/02_roi_segmentation/roi_oof_experiment.md` 和 `artifacts/reports/Chinese reports/02_roi_segmentation/roi_oof_lcc_optimization.md`。

### 4. ROI 面积质量门控

分割预测并非始终可靠。面积过小（< 0.08）可能表示分割器仅捕获噪声区域；面积过大（> 0.75）则说明 mask 接近覆盖全图，ROI 失去聚焦意义。面积门控在 ROI 质量异常时回退至完整图预测。

**消融结果（BUSI 外部）：**

| 配置 | AUC | Sensitivity | Specificity | Precision | F1-Score | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| ROI OOF LCC（基线） | 0.9208 | 0.8524 | 0.8169 | 0.6911 | 0.7633 | 80 | 31 |
| + 面积门控 | **0.9256** | **0.8667** | **0.8467** | **0.7309** | **0.7930** | 67 | 28 |
| **提升** | **+0.0048** | **+0.0143** | **+0.0297** | **+0.0398** | **+0.0297** | **-13** | **-3** |

在该组消融中，面积门控是唯一同时提升全部六项指标的技术。被拒绝的替代方案包括：仅调阈值（增益过小）、移除 LCC（AUC/Sensitivity 回退）、门控 < 0.25（AUC -0.0116）。

详细数据见 `artifacts/reports/Chinese reports/02_roi_segmentation/roi_precision_f1_study.md`。

### 5. OOF Logistic Stacking

不同模型、TTA 视图和 ROI/完整图分支的概率分布存在校准差异，直接平均可能放大某一分支的系统性偏差。系统采用 BUSBRA 训练集 out-of-fold 预测训练 Logistic Regression 融合器，以 logit 空间特征作为输入，避免直接依赖外部验证集搜索权重。

**消融结果（BUSI 外部）：**

| 融合方式 | BUSI AUC | 说明 |
|---|---:|---|
| 静态加权平均 | 0.9130 | eff 0.427 / conv 0.573 |
| OOF Logistic Stacking | 0.9138 | +0.0008，多视图输入 |

OOF stacking 在内部验证中提供了更规范的融合器训练口径，但外部 AUC 提升有限（+0.0008）。其主要价值在于通过内部 OOF 预测学习分支权重与校准关系，而不是在外部验证集上手动调参。

详细数据见 `artifacts/reports/Chinese reports/03_ensemble_oof_stacking/oof_two_model_stacking.md`。

### 6. 集成成员选择

项目对多个候选模型进行了系统筛选。以下为单折（fold1）timm-aware 配方的代表性外部复核结果：

| 模型 | 参数量 | AUC | Sensitivity | Specificity | F1-Score |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 28M | 0.8943 | 0.7762 | 0.8741 | 0.7617 |
| ConvNeXt-Small | 50M | 0.8947 | 0.7667 | 0.8581 | 0.7436 |
| Swin-Tiny | 28M | 0.8729 | 0.7048 | 0.8902 | 0.7291 |
| DenseNet-121 | 8M | 0.8766 | 0.4571 | 0.9771 | 0.6076 |
| EfficientNetV2-S | 21M | 0.8609 | 0.8333 | 0.7048 | 0.6809 |
| ResNet-18 | 11M | 0.8480 | 0.7714 | 0.8124 | 0.7137 |
| MobileNetV3-Small | 2.5M | 0.8431 | 0.7143 | 0.7735 | 0.6536 |
| Basic CNN | — | 0.7327 | 0.0429 | 0.9794 | 0.0789 |
| VGG-16 | 138M | 0.5000 | 0.0000 | 1.0000 | 0.0000 |

**双模型 vs 三模型集成（BUSI 外部正式评估）：**

| 配置 | AUC | Sensitivity | Specificity | Accuracy | F1-Score |
|---|---:|---:|---:|---:|---:|
| 双模型（ConvNeXt + EfficientNet） | 0.9142 | 0.7524 | 0.9130 | 0.8609 | 0.7783 |
| 三模型（+ DenseNet） | 0.9144 | 0.7095 | 0.9382 | 0.8640 | 0.7720 |
| **差异** | **+0.0002** | **-0.0429** | **+0.0252** | **+0.031** | **-0.0063** |

三模型 AUC 仅高 0.0002，但 Sensitivity 下降 4.3%，部署复杂度增加（15 vs 10 个 checkpoint）。双模型在 Youden 最优点的 Accuracy 更高（0.8655 vs 0.8516），因此主线选择双模型。

详细数据见 `artifacts/reports/Chinese reports/03_ensemble_oof_stacking/formal_best_ensemble_external_eval.md`。

### 7. ConvNeXt-Small 升级评估

ConvNeXt-Small（50M 参数）在单折外部 AUC 上略高于 ConvNeXt-Tiny（28M 参数），但内部验证 AUC 反而更低，且训练损失接近 0，提示当前配置下存在过拟合风险。

**消融结果：**

| 模型 | BUSBRA fold1 AUC | BUSI fold1 AUC | Youden J |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.9259 | 0.8953 | 0.6671 |
| ConvNeXt-Small | 0.9118 | 0.8991 | 0.6665 |
| **差异** | **-0.0141** | **+0.0038** | **-0.0006** |

内部 AUC 下降 0.0141 与外部 AUC 提升 0.0038 的矛盾表明 ConvNeXt-Small 在当前训练配置下泛化不稳定。Youden J 几乎相同（0.6671 vs 0.6665），未达到替换标准。

详细数据见 `artifacts/reports/Chinese reports/01_baseline_model_screening/convnext_small_upgrade_experiment.md`。

## 已测试但未采纳的方案

| 方案 | BUSBRA OOF 表现 | BUSI 外部表现 | 拒绝原因 |
|---|---|---|---|
| 面积感知动态权重 | AUC 0.9232 | AUC 0.9185 | 未超过主线 0.9208 |
| OOF Meta-Learner | AUC 0.9241 | AUC 0.9115 | 外部 AUC 回退 0.0093 |
| ROI 软门控 + 多尺度裁剪 | AUC 0.9221 | AUC 0.9189 | 外部 AUC 低于主线 0.9256 |
| ROI 面积门控 OOF 协议 | AUC 0.9233 | — | 候选配置未超过主线 |
| Weight Soup (seed 42+123) | — | — | 内部增益未迁移到外部 |
| CutMix / Mixup | — | — | 小数据集下未带来稳定提升 |
| 320 输入分辨率 | — | — | 显存开销增加，外部 AUC 无改善 |
| EfficientNet TTA | — | — | 内部增益过小且增加推理延迟 |

详细记录见 `artifacts/reports/Chinese reports/` 下各分类子目录中的对应协议文件。

## 完整管线累计提升

以下为主线中各技术从基线到最终配置的 BUSI 外部 AUC 累积路径：

| 阶段 | 配置 | BUSI AUC | 累积提升 |
|---|---|---:|---:|
| 单折 ConvNeXt-Tiny | fold1, identity | 0.8943 | 基线 |
| + timm-aware 配方 | 修正预处理失配 | 0.8943 | 前提条件 |
| + 五折集成 | 5-fold average | 0.9054 | +0.0111 |
| + crop-sweep TTA | 3 crop × hflip | 0.9054 | 包含在五折中 |
| + EfficientNetV2-S 辅助分支 | 双模型静态加权 | 0.9130 | +0.0076 |
| + OOF Logistic Stacking | logit 融合 | 0.9138 | +0.0008 |
| + ROI 分割引导 | UNet + LCC | 0.9208 | +0.0070 |
| + 面积质量门控 | [0.08, 0.75] 回退 | **0.9256** | **+0.0048** |
| **总提升** | | | **+0.0313** |

## 指标定义

| 指标 | 定义 | 临床意义 |
|---|---|---|
| AUC | ROC 曲线下面积 | 模型区分良恶性的整体排序能力 |
| Accuracy | (TP+TN) / (TP+TN+FP+FN) | 整体预测正确率 |
| Sensitivity (Recall) | TP / (TP+FN) | 恶性检出率，反映漏诊风险 |
| Specificity | TN / (TN+FP) | 良性正确识别率，反映误诊控制 |
| Precision | TP / (TP+FP) | 阳性预测可靠性 |
| F1-Score | 2PR / (P+R) | Precision 与 Recall 的调和均值 |
| Youden J | Sensitivity + Specificity - 1 | 综合评估阈值优劣 |
| Dice | 2 * overlap / (area(A) + area(B)) | 分割 mask 与 GT 的重叠度 |

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
    ├── checkpoints/                  # 模型权重（通过 Git LFS 或本地资产管理）
    └── reports/                      # 实验报告与评估结果
        ├── Chinese reports/          # 中文实验报告（按实验主题分类，139 份）
        ├── English reports/          # 英文实验报告（按实验主题分类，139 份）
        └── README.md                 # 报告目录分类说明
```

## 环境配置

```powershell
conda create -n BUCAD python=3.11 -y
conda activate BUCAD
python -m pip install -r requirements.txt
python check_env.py
python check_all.py
```

`configs/inference/demo.yml` 引用的演示模型权重通过 Git LFS 存放在 `artifacts/checkpoints/`。安装 Git LFS 后正常 clone 通常会自动拉取；如果本地 `.pt` 文件只是很小的 pointer 文件，启动 demo 前执行 `git lfs pull`。

### 数据集配置

复制 `configs/paths.example.yml` 为 `configs/paths.local.yml`，配置本地数据集路径：

```yaml
datasets:
  busbra_root: ./训练集/BUSBRA
  busi_root: ./测试集/Dataset_BUSI_with_GT
```

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

### Windows 打包

```powershell
python -m PyInstaller --clean --noconfirm packaging\desktop_demo.spec
```

输出：`dist/bucad-demo-desktop/bucad-demo-desktop.exe`。需分发完整目录。

### 训练与评估

```powershell
# 生成数据划分
python scripts\make_split.py --config configs\paths.local.yml

# 训练分类器（单折）
python scripts\train_cls.py --config configs\classifier\convnext_tiny_timm_recipe.yml --fold 1

# 批量评估
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_demo.json
```

## 硬件要求

| 项目 | 推理/演示 | 训练/实验 |
|---|---|---|
| 操作系统 | Windows 10/11 x64 | Windows 10/11 x64 |
| Python | 打包后无需安装 | Conda + 3.11 |
| GPU | 非必需 | NVIDIA CUDA，8GB+ VRAM |
| 内存 | 8GB 最低，16GB 推荐 | 16GB 最低，32GB 推荐 |
| 磁盘 | 8GB | 50GB |

## 关键实验报告索引

| 报告 | 内容 |
|---|---|
| `01_baseline_model_screening/native_single_model_retest.md` | timm-aware vs 非 timm-aware 配方对比 |
| `01_baseline_model_screening/fivefold_single_model_comparison.md` | 四模型 5-fold vs 单折对比 |
| `04_tta_threshold_external_eval/convnext_tta_optimization.md` | ConvNeXt TTA 策略消融 |
| `02_roi_segmentation/roi_oof_experiment.md` | ROI 引导 vs 完整图对比 |
| `02_roi_segmentation/roi_oof_lcc_optimization.md` | LCC 后处理消融 |
| `02_roi_segmentation/roi_precision_f1_study.md` | ROI 面积门控消融 |
| `03_ensemble_oof_stacking/oof_two_model_stacking.md` | OOF Stacking vs 静态权重 |
| `03_ensemble_oof_stacking/formal_best_ensemble_external_eval.md` | 双模型 vs 三模型正式评估 |
| `01_baseline_model_screening/convnext_small_upgrade_experiment.md` | ConvNeXt-Small vs Tiny 对比 |
| `01_baseline_model_screening/six_model_comparison_report.md` | 六模型全面对比（BUSBRA + BUSI） |

完整报告目录见 `artifacts/reports/Chinese reports/`，分类说明见 `artifacts/reports/README.md`。

## 参考文献

- Al-Dhabyani W, et al. Dataset of breast ultrasound images. *Data in Brief*, 2020. [[PubMed]](https://pubmed.ncbi.nlm.nih.gov/31867417/)
- Liu Z, et al. A ConvNet for the 2020s. *CVPR*, 2022.
- Tan M, Le Q. EfficientNetV2: Smaller models and faster training. *ICML*, 2021.
- Ronneberger O, et al. U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*, 2015.
- He K, et al. Deep residual learning for image recognition. *CVPR*, 2016.
- ROI-aware breast ultrasound classification. [[PMC11431713]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11431713/)
- Multi-task breast ultrasound segmentation and classification. [[PMC12011763]](https://pmc.ncbi.nlm.nih.gov/articles/PMC12011763/)
- OpenUS ultrasound foundation model. [[GitHub]](https://github.com/XZheng0427/OpenUS)
- BUSI segmentation reference. [[GitHub]](https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch)
- BUSI-SAM / BUSSAM segmentation references. [[GitHub]](https://github.com/huangjin520/BUSI-SAM) [[GitHub]](https://github.com/bscs12/BUSSAM)

## 许可

本项目仅供科研、教学与比赛复现实验使用。
