<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->
# 优化尝试记录 2026-04-27

## 边界条件

- 候选方案的筛选与训练变更仅使用 BUSBRA 内部 OOF/训练数据。
- BUSI 仅在 model-zoo 候选方案冻结后使用。
- 本报告中任何失败的候选方案均不应合并至 `configs/inference/demo.yml`。

## 基线

当前 demo 主线配置：

- 配置文件：`configs/inference/demo.yml`
- BUSI 外部 AUC：`0.9256`
- BUSI Sensitivity：`0.8667`
- BUSI Specificity：`0.8467`
- BUSI Precision：`0.7309`
- BUSI F1：`0.7930`
- BUSI 混淆矩阵：TN `370` / FP `67` / FN `28` / TP `182`

## 尝试 1：Model-Zoo OOF 候选方案

BUSBRA 内部 OOF 筛选结果：

- 配置文件：`configs/inference/demo_model_zoo_oof_candidate.yml`
- 全图权重：
  - EfficientNetV2-S：`0.3057`
  - ConvNeXt-Tiny：`0.0321`
  - DenseNet121：`0.2315`
  - ConvNeXt-Small：`0.0400`
  - Swin-Tiny：`0.3907`
- OOF 嵌套 AUC：`0.9352`
- OOF 选定阈值：`0.41`
- OOF Sensitivity：`0.8402`
- OOF Specificity：`0.8927`
- OOF Precision：`0.7895`
- OOF F1：`0.8140`

BUSI 冻结外部评估：

- 报告路径：`artifacts/reports/busi_demo_model_zoo_oof_candidate_eval.json`
- AUC：`0.9213`
- Sensitivity：`0.7952`
- Specificity：`0.8787`
- Precision：`0.7591`
- F1：`0.7767`
- 混淆矩阵：TN `384` / FP `53` / FN `43` / TP `167`

决策：放弃。尽管 BUSBRA OOF 有较大提升，外部 AUC 和 Sensitivity 均下降。这与 model-zoo/stacker 对 BUSBRA OOF 分布过拟合的判断一致。

## 尝试 2：困难样本 Focal 重训练

变更内容：

- 在 `src/engine/train_cls.py` 中新增 `loss: focal` 支持。
- 通过 `training.sample_weight_path` 新增样本权重支持。
- 基于 `artifacts/reports/busbra_oof_demo_error_analysis.csv` 构建困难样本权重。
- 配置文件：`configs/classifier/convnext_tiny_hard_oof_focal.yml`

Fold1 结果：

- 原始 ConvNeXt-Tiny fold1 最佳 AUC：`0.9259`
- Hard/focal fold1 最佳 AUC：`0.9051`
- Hard/focal fold1 Sensitivity：`0.7869`
- Hard/focal fold1 Specificity：`0.8221`
- Hard/focal fold1 F1：`0.7300`

决策：在 fold1 后停止。该设置过度聚焦困难样本，降低了排序质量。

## 尝试 3：温和困难样本重训练

变更内容：

- 使用交叉熵损失而非 focal loss。
- 保持基线数据增强。
- 使用温和的 OOF 错误样本权重：
  - FN：`1.5-1.8`
  - FP：`1.15-1.2`
- 配置文件：`configs/classifier/convnext_tiny_hard_oof_mild.yml`

Fold1 结果：

- 原始 ConvNeXt-Tiny fold1 最佳 AUC：`0.9259`
- 温和困难样本 fold1 最佳 AUC：`0.9086`
- 温和困难样本 fold1 Sensitivity：`0.7213`
- 温和困难样本 fold1 Specificity：`0.9051`
- 温和困难样本 fold1 F1：`0.7521`

决策：在 fold1 后停止。温和加权仍低于原始 fold1 表现，因此不值得进行完整的 5-fold 训练。

## 当前结论

- 当前 demo 仍为最佳可部署主线。
- 最大的实际风险是通过元模型或不可迁移的样本权重对 BUSBRA OOF 过拟合。
- 下一个高价值方向应为更保守的训练实验，例如针对已表现优秀的 ConvNeXt-Tiny 方案进行 seed/fold 多样性实验，或采用真正的嵌套 OOF 协议，在不重复使用相同 OOF 预测的前提下选择模型族。

## 尝试 4：ROI 软门控 + 多尺度裁剪

BUSBRA 内部 OOF 筛选结果：

- 报告路径：`artifacts/reports/roi_soft_gate_multiscale_protocol.md`
- 候选方案：margins `[0.2, 0.35]`，weights `[0.35, 0.65]`，min area `0.15`，max area `1.01`，ramp `0.03`，max weight `1.0`
- OOF AUC：`0.9221`
- OOF Sensitivity：`0.8402`
- OOF Specificity：`0.8691`
- OOF Precision：`0.7544`
- OOF F1：`0.7950`

BUSI 冻结外部评估：

- AUC：`0.9189`
- Sensitivity：`0.8429`
- Specificity：`0.8375`
- Precision：`0.7137`
- F1：`0.7729`
- 混淆矩阵：TN `366` / FP `71` / FN `33` / TP `177`

决策：放弃。软门控减少了 BUSBRA OOF 误报，但未能迁移至 BUSI。

## 尝试 5：ConvNeXt-Tiny Seed 多样性

变更内容：

- 补全缺失的 `convnext_tiny_timm_recipe_seed123` 中 fold `1/2/3/5`。
- 新增 `scripts/run_seed_diversity_oof_protocol.py`。
- 仅使用 BUSBRA OOF 进行筛选，搜索范围较窄：在现有 ConvNeXt 家族权重中分配 seed42 和 seed123 的比例，保持 EfficientNetV2-S 权重和当前 ROI area gate 不变。

BUSBRA 内部 OOF 筛选结果：

- 配置文件：`configs/inference/demo_seed_diversity_oof_candidate.yml`
- Seed123 占比：`0.67`
- ROI stack 混合权重：`0.75`
- OOF AUC：`0.9249`
- OOF Sensitivity：`0.8402`
- OOF Specificity：`0.8667`
- OOF Precision：`0.7511`
- OOF F1：`0.7932`
- OOF 混淆矩阵：TN `1099` / FP `169` / FN `97` / TP `510`

BUSI 冻结外部评估：

- 报告路径：`artifacts/reports/busi_demo_seed_diversity_oof_candidate_eval.json`
- AUC：`0.9250`
- Sensitivity：`0.8333`
- Specificity：`0.8581`
- Precision：`0.7384`
- F1：`0.7830`
- 混淆矩阵：TN `375` / FP `62` / FN `35` / TP `175`

决策：放弃。外部 AUC 略低于当前 demo，Sensitivity 下降过多。

## 尝试 6：ConvNeXt-Tiny 权重汤（Weight Soup）

变更内容：

- 新增 `scripts/build_convnext_seed_soup.py`。
- 新增 `scripts/run_convnext_soup_oof_protocol.py`。
- 构建同折 seed42/seed123 线性权重平均，alpha 为 `0.50`。

BUSBRA 内部 OOF 筛选结果：

- 配置文件：`configs/inference/demo_convnext_soup_oof_candidate.yml`
- ROI stack 混合权重：`0.75`
- OOF AUC：`0.9232`
- OOF Sensitivity：`0.8501`
- OOF Specificity：`0.8462`
- OOF Precision：`0.7257`
- OOF F1：`0.7830`
- OOF 混淆矩阵：TN `1073` / FP `195` / FN `91` / TP `516`

BUSI 冻结外部评估：

- 报告路径：`artifacts/reports/busi_demo_convnext_soup_oof_candidate_eval.json`
- AUC：`0.9225`
- Sensitivity：`0.8619`
- Specificity：`0.8238`
- Precision：`0.7016`
- F1：`0.7735`
- 混淆矩阵：TN `360` / FP `77` / FN `29` / TP `181`

决策：放弃。与概率级 seed 多样性相比，权重平均降低了运行时开销，但外部 AUC 和 F1 均下降。

## 尝试 7：轻度正则化重训练

变更内容：

- 新增 `configs/classifier/convnext_tiny_timm_recipe_regularized.yml`。
- 尝试轻度旋转、亮度/对比度增强、尺度抖动、标签平滑和更高的权重衰减。

Fold1 内部筛选：

- 原始 ConvNeXt-Tiny fold1 最佳 AUC：`0.9259`
- 正则化 fold1 最佳 AUC：`0.9096`
- 正则化 fold1 Sensitivity：`0.8443`
- 正则化 fold1 Specificity：`0.8656`
- 正则化 fold1 F1：`0.7954`

决策：在 fold1 后停止。AUC 与原始 fold1 差距过大，不值得进行完整的 5-fold 训练。

## 尝试 8：CutMix 重训练

变更内容：

- 在 `src/engine/train_cls.py` 中新增可选的 `training.mixup_alpha`、`training.cutmix_alpha` 和 `training.mix_probability` 支持。
- 新增 `configs/classifier/convnext_tiny_timm_recipe_cutmix.yml`。

Fold1 内部筛选：

- 原始 ConvNeXt-Tiny fold1 最佳 AUC：`0.9259`
- CutMix fold1 最佳 AUC：`0.9145`
- CutMix fold1 Sensitivity：`0.7459`
- CutMix fold1 Specificity：`0.9289`
- CutMix fold1 F1：`0.7879`

决策：在 fold1 后停止。CutMix 提高了 Specificity 但损害了排序质量和 Sensitivity。

## 尝试 9：EfficientNetV2-S TTA

变更内容：

- 新增 `scripts/run_efficientnet_tta_oof_protocol.py`。
- 测试 EfficientNetV2-S 的 hflip 和 crop-sweep 测试时增强（TTA），保持分类器权重不变。

BUSBRA 内部 OOF 筛选结果：

- 配置文件：`configs/inference/demo_efficientnet_tta_oof_candidate.yml`
- EfficientNet 视图：`eff_hflip`
- ROI stack 混合权重：`0.75`
- OOF AUC：`0.9190`
- OOF Sensitivity：`0.8402`
- OOF Specificity：`0.8438`
- OOF Precision：`0.7203`
- OOF F1：`0.7757`

决策：不运行 BUSI。相较于当前 demo OOF 的内部增益过小，额外的推理成本不值得，不值得再消耗一次外部验证。

## 尝试 10：ConvNeXt-Tiny 320 输入

变更内容：

- 新增 `configs/classifier/convnext_tiny_timm_recipe_320.yml`。
- 保持基线 ConvNeXt 配方，但将分类器输入提升至 `320`，batch size 为 `6`，学习率为 `3e-5`。

Fold1 内部筛选：

- 原始 ConvNeXt-Tiny fold1 最佳 AUC：`0.9259`
- ConvNeXt-Tiny 320 fold1 最佳 AUC：`0.9210`
- ConvNeXt-Tiny 320 fold1 Sensitivity：`0.7049`
- ConvNeXt-Tiny 320 fold1 Specificity：`0.9289`
- ConvNeXt-Tiny 320 fold1 F1：`0.7611`

决策：在 fold1 后停止。更高分辨率提高了 Specificity，但未能超越原始 fold1 的排序质量，且降低了 Sensitivity。

## 更新后的结论

- 所有新的冻结候选方案均未能提升当前 BUSI AUC `0.9256`，因此 `configs/inference/demo.yml` 保持不变。
- 最强的内部增益来自 ConvNeXt seed 多样性，但未能成功迁移至外部数据；这进一步说明 BUSBRA OOF 在额外集成复杂度方面目前过于乐观。
- 困难样本加权、正则化、CutMix、model-zoo 堆叠、seed 多样性、权重汤、软 ROI 门控、EfficientNet TTA 以及朴素的 320 输入训练，在当前证据下均为不良合并候选。
- 下一次尝试应避免更多的后验 OOF 融合和简单的增强变更。更好的下一步方向是采用病灶感知训练目标或验证稳定的预训练/域适应策略，并在任何 BUSI 评估之前逐折筛选。
