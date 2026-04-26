# BUCAD 中文说明

BUCAD（Breast Ultrasound Computer-Aided Diagnosis）是一个面向乳腺超声肿瘤良恶性分类与可视化的科研原型项目。项目覆盖数据划分、分类模型训练、BUSI 外部评估、病灶定位可视化、Grad-CAM 风格解释、本地 Gradio 网页演示和 Windows 打包。

> BUCAD 是科研和原型系统，不是临床诊断产品，不能替代医生判断。

## 项目功能

输入一张乳腺超声图像后，系统可以：

- 输出良性/恶性概率；
- 根据可配置阈值给出最终判定；
- 输出置信度和边界样本提醒；
- 在分割权重可用时生成病灶定位叠加图；
- 从主分类模型生成 Grad-CAM 风格热力图；
- 对 BUSI 数据集进行批量外部评估；
- 启动本地 Gradio 网页界面进行单图演示。

## 当前演示模型

当前主线是 `ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF LCC`，运行配置在 `configs/inference/demo.yml`。

- **主模型分支**：`ConvNeXt-Tiny` 五折 checkpoint，集成权重 `0.573`，使用 timm-aware 预处理和 crop-sweep TTA；网页 UI 与 Grad-CAM 解释也以它作为主模型。
- **辅助模型分支**：`EfficientNetV2-S` 五折 checkpoint，集成权重 `0.427`，使用 CLAHE 预处理和 identity TTA；它用于补充 ConvNeXt-Tiny 的错误分布，同时比继续加入 DenseNet/Swin 更轻量。
- **ROI 分支**：`segmenter_fold1.pt` 预测病灶 mask；根据 BUSBRA 分割验证结果使用 `mask_threshold=0.40`、`margin_ratio=0.35` 和最大连通域裁剪。
- **OOF 融合**：系统会同时计算完整图概率和 ROI 概率，再输入 BUSBRA OOF 训练出的 logit logistic stacker 得到最终恶性概率。
- **判定阈值**：`0.550`，来自 BUSBRA ROI out-of-fold 预测，不使用 BUSI 选择阈值。
- **数据边界**：BUSBRA 用于训练、内部验证、OOF 模型选择、ROI 参数选择和阈值选择；BUSI 只在配置固定后用于外部评估。

当前 demo 配置在 BUSI 外部评估上的结果：

| 模型 | AUC | 阈值 | Sensitivity | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny + EfficientNetV2-S + ROI OOF Stacking + LCC ROI | 0.9208 | 0.550 | 0.8524 | 0.8169 | 0.8284 |

保留的最强非 ROI 对比基准：

| 模型 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt-Tiny + EfficientNetV2-S + DenseNet121 优化集成 | 0.9162 | 0.453 | 0.7476 | 0.9291 | 0.8702 | 报告对比 / 上限参考 |

三模型方案作为非 ROI 对照保留；当前 ROI 引导的两模型主线 AUC 更高，同时复杂度低于 15 个 checkpoint 的三模型集成。

## 为什么选择 ConvNeXt-Tiny 和 EfficientNetV2-S

当前 demo 候选方案记录在 `artifacts/reports/` 下的实验报告中。报告里的 BUSI 指标只能作为配置固定后的外部观察，不能作为训练、调参、模型族选择、集成权重选择或阈值选择的唯一依据。

- `fivefold_single_model_comparison.md`：记录四个五折单模型在 BUSI 上的外部观察表现；ConvNeXt-Tiny 是当前观察到的最强轻量单模型候选。
- `convnext_tta_optimization.md`：ConvNeXt-Tiny 对 timm-aware 预处理和 crop-sweep TTA 收益明显，说明它适合作为主模型继续优化。
- `formal_best_ensemble_external_eval.md`：EfficientNetV2-S + ConvNeXt-Tiny 在正式外部评估中已经接近三模型结果。
- `ensemble_tta_threshold_tuning.md`：保留 ConvNeXt crop-sweep TTA、去掉 CNN 分支 hflip 后，两模型 AUC 提升到 `0.9151`。
- `roi_oof_experiment.md` 和 `roi_oof_lcc_optimization.md`：BUSBRA OOF 训练出的 ROI Stacking 叠加 BUSBRA 选择出的 LCC mask 后处理，将锁定 BUSI 外部 AUC 提升到 `0.9208`。
- `four_model_ensemble_weight_search.md`：加入 Swin-Tiny 后没有提升最高 AUC；AUC 最优点会把 Swin 权重压到 `0.000`。

工程解释：当前 demo 暂时保留轻量两模型候选，但比赛最终模型必须先根据 BUSBRA OOF/内部验证锁定，再进行最终外部评估。

- **ConvNeXt-Tiny 做主模型**：它是当前最强单模型家族，且放在 demo 的第一分支后，Grad-CAM 解释来自 ConvNeXt 主模型。
- **EfficientNetV2-S 做辅助模型**：它和 ConvNeXt 的错误分布有互补性，可以提高集成排序能力，复杂度又低于继续加入 DenseNet/Swin。
- **DenseNet121 保留为基准**：三模型结果略强，但增加五个 checkpoint，且当前优化阈值下 Sensitivity 不如两模型。
- **Swin-Tiny 不进入最终 demo**：混合权重搜索显示它对最高 AUC 的贡献很低或为零。

## 优化内容

所有优化都必须遵守数据边界：BUSBRA 用于训练、内部验证、OOF 模型选择和阈值选择；BUSI 只在配置固定后用于外部评估，不参与训练、调参、权重选择或阈值选择。

主要优化点：

1. **异构预处理**：每个集成成员可以独立设置图像尺寸、CLAHE、归一化、插值、crop 比例和 TTA。
2. **ConvNeXt crop-sweep TTA**：ConvNeXt 使用 ImageNet mean/std、bicubic 插值和 `crop_pct=0.90/0.95/1.00`，每个 crop 同时评估原图和水平翻转。
3. **EfficientNet identity 分支**：EfficientNetV2-S 使用 CLAHE、area 插值、不继承 ConvNeXt 的 mean/std，并只保留 identity TTA，避免跨模型预处理串扰。
4. **ROI OOF Stacking**：最终 demo 同时使用完整图概率和 ROI 概率，由 BUSBRA OOF 训练出的逻辑融合器输出最终概率。
5. **ROI mask 后处理**：根据 BUSBRA 分割验证结果，最终 ROI 使用 `0.40` mask 阈值、最大连通域裁剪和 `margin_ratio=0.35`。
6. **OOF 阈值选择**：最终 ROI stack 阈值为 `0.550`，来自 BUSBRA out-of-fold 预测，避免在 BUSI 测试集上调阈值。
7. **demo 对齐**：`demo.yml` 将 ConvNeXt-Tiny 放在第一分支，网页 UI 也声明 ConvNeXt-Tiny 是主模型。

## 项目目录

- `configs/`：路径、分类器、分割器和推理配置。
- `src/datasets/`：BUSBRA/BUSI 数据读取与 split 支持。
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

数据边界：

- BUSBRA 用于训练和内部验证。
- BUSI 只用于外部评估和 demo 验证。
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

## 启动网页 demo

```powershell
conda activate BUCAD
python app\main.py
```

程序会读取 `configs/inference/demo.yml` 并启动本地 Gradio 网页界面。当前 demo 声明 `ConvNeXt-Tiny` 为主模型，`EfficientNetV2-S` 为辅助集成分支。

## BUSI 外部评估

```powershell
python scripts\eval_busi.py --config configs\inference\demo.yml --output artifacts\reports\busi_demo_convnext_effnet.json
```

当前期望指标接近：

| AUC | 阈值 | Sensitivity | Specificity | Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.9208 | 0.550 | 0.8524 | 0.8169 | 0.8284 |

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

## 重要报告

项目相关的关键报告：

- `artifacts/reports/fivefold_single_model_comparison.md`
- `artifacts/reports/fold1_single_model_baseline_comparison.md`
- `artifacts/reports/convnext_tta_optimization.md`
- `artifacts/reports/formal_best_ensemble_external_eval.md`
- `artifacts/reports/ensemble_tta_threshold_tuning.md`
- `artifacts/reports/four_model_ensemble_weight_search.md`
- `artifacts/reports/swin_tiny_5fold_experiment.md`
- `artifacts/reports/training_recipe_audit.md`
- `artifacts/reports/literature_guided_optimization.md`
- `artifacts/reports/final_validation.md`

## 参考与致谢

本项目在设计和优化过程中参考了公开乳腺超声数据集、医学影像开源项目，以及分类、分割、ROI 感知诊断、多任务学习和超声基础模型相关研究。

- BUSI 数据集：[Dataset of breast ultrasound images](https://pubmed.ncbi.nlm.nih.gov/31867417/)
- 乳腺超声病灶区域感知分类研究：[PMC11431713](https://pmc.ncbi.nlm.nih.gov/articles/PMC11431713/)
- 乳腺超声分割与分类多任务研究：[PMC12011763](https://pmc.ncbi.nlm.nih.gov/articles/PMC12011763/)
- OpenUS 超声基础模型：[XZheng0427/OpenUS](https://github.com/XZheng0427/OpenUS)
- BUSI 分割参考项目：[tqxli/breast_ultrasound_lesion_segmentation_PyTorch](https://github.com/tqxli/breast_ultrasound_lesion_segmentation_PyTorch)
- BUSI-SAM / SAM 风格分割参考：[huangjin520/BUSI-SAM](https://github.com/huangjin520/BUSI-SAM)、[bscs12/BUSSAM](https://github.com/bscs12/BUSSAM)

感谢上述数据集、论文和开源项目的作者与维护者。他们的工作为 BUCAD 的数据处理、模型对比、分割可视化、混合集成设计和后续 ROI 感知优化提供了重要参考。
