# SonoGloReNet-T fold1 性能下降诊断

日期：2026-06-18

## 结论摘要

- 当前性能下降不是单纯阈值问题，AUC 在内部验证和 BUSI 外部评估上都下降，说明排序能力本身变弱。
- 主要退化形式是 benign 样本被更频繁、且更高置信度地推向 malignant。
- `基线正确 -> SonoGloReNet 错误` 共 54 例，其中 45 例是 benign -> malignant，9 例是 malignant -> benign。
- 这说明当前版本更偏向激进召回，代价是 specificity 和 precision 明显下降。

## 量化证据

- BUSI AUC：基线 `0.8926`，SonoGloReNet `0.8782`，下降约 `0.0144`
- 内部验证 AUC：基线 `0.9259`，SonoGloReNet `0.9096`，下降约 `0.0164`
- BUSI 默认阈值下 false positive：基线 `57`，SonoGloReNet `89`
- BUSI 默认阈值下 false negative：基线 `47`，SonoGloReNet `41`
- benign 恶性概率均值：基线 `0.1412`，SonoGloReNet `0.2045`
- malignant 恶性概率均值：基线 `0.7627`，SonoGloReNet `0.8023`
- benign 样本中落在高风险区间 `[0.9, 1.0]` 的数量：基线 `25`，SonoGloReNet `61`

## 错判病例分析

### 新增误判

- 新增误判总数：`54`
- 其中 benign -> malignant：`45`
- 其中 malignant -> benign：`9`

### 代表性病例

- `benign (414)`：基线 `0.0017` -> SonoGloReNet `0.9968`
- `benign (104)`：基线 `0.0097` -> SonoGloReNet `0.9857`
- `benign (233)`：基线 `0.0263` -> SonoGloReNet `0.9991`
- `benign (63)`：基线 `0.0170` -> SonoGloReNet `0.9812`
- `malignant (68)`：基线 `0.9981` -> SonoGloReNet `0.0736`
- `malignant (88)`：基线 `0.9141` -> SonoGloReNet `0.0003`

### 结构性模式

- 新增误判以 benign 被高置信度推成 malignant 为主。
- 新增误判病例的平均病灶面积比例约为 `0.1100`，高于修复误判病例的 `0.0701`。
- 新增高置信度 false positive 数量为 `27`，说明问题不是只在阈值附近轻微波动，存在明显的过度放大。

完整病例明细见：

- `artifacts/reports/generated/sonoglore_fold1_worsened_cases.csv`
- `artifacts/reports/generated/sonoglore_vs_convnext_fold1_case_diff.csv`

## 最可疑的问题点

### 1. `GeM + GELU` 组合存在实现风险

当前 `src/models/classifier.py` 中：

- [src/models/classifier.py](C:/Users/876762330/Desktop/projects/Agent/src/models/classifier.py:143) 到 [src/models/classifier.py](C:/Users/876762330/Desktop/projects/Agent/src/models/classifier.py:152)
- [src/models/classifier.py](C:/Users/876762330/Desktop/projects/Agent/src/models/classifier.py:56) 到 [src/models/classifier.py](C:/Users/876762330/Desktop/projects/Agent/src/models/classifier.py:60)

每个 stage 的投影分支使用 `1x1 Conv -> GELU -> GeM`，而 GeM 内部会先对输入做 `clamp(min=eps)`。

这会带来一个问题：

- `GELU` 会产生负值
- 负值进入 GeM 之前会被大量截断为极小正数
- 特征分布会被强烈压扁

这和当前现象一致：大量 benign 被推到极高 malignant 概率。

### 2. 新头没有复用原始 ConvNeXt 最终语义整理路径

当前实现直接使用 `features_only=True` 的 stage 输出做分类，没有复用 timm 原始分类头中的最终 `norm/head` 路径。

这意味着：

- 当前模型的最后特征语义整理方式与原始 backbone 已验证的路径不同
- 多尺度分支可能引入更强的纹理偏置
- 对 benign 复杂纹理的误报会更重

### 3. 正则化不足

训练报告显示：

- epoch 19 之后训练 loss 已接近 0
- 但验证 AUC 没有同步提升

这说明新头容量增大后，已经出现明显记忆训练集倾向。

## 建议优先修复顺序

1. 先改 pooling 分支，把 `GELU` 改成 `ReLU` 或 `Softplus` 后再接 GeM
2. 做一版消融：去掉 `stage4 attention`，只保留多尺度 pooling
3. 提高正则强度：`drop_path_rate`、`label_smoothing`、`MixUp/CutMix`
4. 若仍然不稳，改成只融合 `stage3 + stage4`，降低浅层纹理干扰

## 补充实验：仅替换为 ReLU 后的结果

我额外做了一轮最小修复实验：

- 训练配置：`configs/classifier/sonoglore_convnext_tiny_relu.yml`
- 推理配置：`configs/inference/sonoglore_convnext_tiny_relu_fold1.yml`

结果如下：

- 内部验证 AUC：`0.9068`
- BUSI 外部 AUC：`0.5711`
- 默认阈值下 Sensitivity：`0.0667`
- 默认阈值下 Specificity：`0.9680`

这说明：

- `GeM + GELU` 确实是可疑点，但它不是唯一根因
- 仅把分支激活改成 `ReLU`，无法单独修复当前结构问题
- 当前多尺度头本身对域外分布非常敏感，存在更深层的表示与校准不稳定

因此更合理的下一步不应只做单点微调，而应优先做结构消融：

1. 去掉 `stage4 attention`
2. 仅保留 `stage4`
3. 仅保留 `stage3 + stage4`
4. 再分别配合轻量正则重新比较
