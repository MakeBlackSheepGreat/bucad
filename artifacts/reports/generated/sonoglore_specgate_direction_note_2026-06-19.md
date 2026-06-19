# SonoGloReNet 下一轮结构方向记录（SpecGate）

日期：2026-06-19

## 1. 当前瓶颈

- `sonoglore_convnext_tiny_grn_eca_asymproj_posw17` 目前仍是外部 BUSI 最强单模型线。
- 但它在 BUSBRA 内部 OOF AUC 仍低于 `convnext_tiny_timm_recipe` 基线。
- 已有结果表明，下列方向没有带来稳定收益：
  - learnable stage fusion
  - 更换 stage4 attention 为 `coordatt / gc / lka`
  - 更强 regularization
  - 结构消融冠军 `ms_noattn__base` 虽通过 fold1 门控，但 5-fold BUSI AUC 仍低于冻结基线

## 2. 论文启发

### ConvNeXt
- Z. Liu et al., *A ConvNet for the 2020s*, CVPR 2022.
- 结论：ConvNeXt 的主干已经足够强，新增模块应尽量轻量、保持 ConvNeXt 的高效卷积归纳偏置。

### FcaNet
- Z. Qin et al., *FcaNet: Frequency Channel Attention Networks*, ICCV 2021 / arXiv:2012.11879.
- 核心点：传统 GAP 只用单个标量表示通道，会丢失频域信息；频域通道压缩可以补充被 GAP 丢掉的成分。

### GFNet
- Y. Rao et al., *Global Filter Networks for Image Classification*, NeurIPS 2021 / arXiv:2107.00645.
- 核心点：频域可以高效建模长程依赖，复杂度优于显式全局注意力。

### MogaNet
- S. Li et al., *MogaNet: Multi-order Gated Aggregation Network*, arXiv:2211.03295.
- 核心点：现代 ConvNet 的瓶颈之一在于上下文聚合和门控表达能力不足，轻量 gated aggregation 往往比盲目堆大卷积更稳。

## 3. 本轮新增结构

已在 `src/models/classifier.py` 新增：

- `SpectralChannelGate2d`
- `use_projection_spectral_gate`
- `projection_spectral_gate_stages`
- `projection_spectral_gate_freq_size`
- `projection_spectral_gate_hidden_dim`
- `projection_spectral_gate_include_spatial`

实现方式：

1. 对每个启用 stage 的 `1x1 projection` 输出做 `rfft2`
2. 取低频幅值块并做固定大小汇聚
3. 与可选空间均值描述子拼接
4. 通过共享小 MLP 生成逐通道 gate
5. 在 GeM pooling 前重标定投影特征

这条线的设计目标：

- 保留当前 `posw17` 主线结构
- 不引入输入分辨率绑定的位置编码
- 不引入重型 Transformer
- 只补足通道压缩阶段对全局频率信息的感知

## 4. 新实验入口

训练配置：
- `configs/classifier/sonoglore_convnext_tiny_grn_eca_asymproj_posw17_specgate.yml`

推理配置：
- `configs/inference/sonoglore_convnext_tiny_grn_eca_asymproj_posw17_specgate_fold1.yml`

## 5. 建议实验顺序

1. 先跑 `fold1`
2. 看三个指标：
   - BUSBRA fold1 AUC
   - BUSI fold1 AUC
   - 若外部不掉，再补 OOF
3. 如果 `fold1` 外部已经明显掉，则停止，不进入 5-fold

## 6. 备注

这次改动本质上是“频域门控增强版的 posw17”，优先级高于继续扫其它 stage attention 变体。
