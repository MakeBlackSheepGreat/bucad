<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->
# 分割器 5-Fold 实验

日期：2026-04-26 22:59:21

## 实验设置

- 配置文件：`configs/segmenter/unet_5fold.yml`
- 架构：U-Net，使用 `resnet18` 编码器
- 图像尺寸：`256`
- 每折训练轮数：`5`
- 输出检查点：`artifacts/checkpoints/segmenter_5fold_fold{fold}.pt`

## Dice 汇总

- 平均 Dice：`0.8675`
- 标准差 Dice：`0.0170`
- 最佳折：`5`（`0.8852`）
- 最弱折：`2`（`0.8383`）

| 折 | Dice | 训练集大小 | 验证集大小 | 检查点路径 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.8593 | 1500 | 375 | `artifacts\checkpoints\segmenter_5fold_fold1.pt` |
| 2 | 0.8383 | 1500 | 375 | `artifacts\checkpoints\segmenter_5fold_fold2.pt` |
| 3 | 0.8747 | 1500 | 375 | `artifacts\checkpoints\segmenter_5fold_fold3.pt` |
| 4 | 0.8800 | 1500 | 375 | `artifacts\checkpoints\segmenter_5fold_fold4.pt` |
| 5 | 0.8852 | 1500 | 375 | `artifacts\checkpoints\segmenter_5fold_fold5.pt` |

## 解读

- 本次运行不会覆盖已冻结的 demo 分割器 `artifacts/checkpoints/segmenter_fold1.pt`。
- 5-fold 检查点可用于 ROI 掩膜稳定性分析及可能的分割集成实验。
- 在替换运行时分割器之前，应评估 5-fold 掩膜是否能在 BUSBRA OOF 上改善 ROI Area Gate 指标，然后进行一次锁定的 BUSI 外部检查。
