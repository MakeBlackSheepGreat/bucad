# U-Net++ ResNet34 主线切换与定制优化小结

## 结论

- 主线 `configs/inference/demo.yml` 已切换为 `U-Net++ ResNet34` 五折分割器，分类主干仍为 `ConvNeXt-Tiny + EfficientNetV2-S + ROI Area Gate`。
- 切换后主线 BUSI 外部复核：AUC `0.923739`，threshold `0.57`，Sensitivity `0.8333`，Specificity `0.8513`，F1 `0.7778`。
- 基于 U-Net++ ResNet34 的局部定制标定没有超过当前 U-Net++ OOF 基准，因此只保留实验报告，不合入。
- Descriptor router 在 BUSBRA OOF 上提高 AUC，但 BUSI 外部复核明显下降，不合入主线。

## 已固定到主线的内容

- `segmenter_checkpoints` 指向：
  - `segmenter_recal_unetplusplus_resnet34_bce_dice_fold1.pt`
  - `segmenter_recal_unetplusplus_resnet34_bce_dice_fold2.pt`
  - `segmenter_recal_unetplusplus_resnet34_bce_dice_fold3.pt`
  - `segmenter_recal_unetplusplus_resnet34_bce_dice_fold4.pt`
  - `segmenter_recal_unetplusplus_resnet34_bce_dice_fold5.pt`
- ROI stacker、阈值和 area gate 使用该分割器的 BUSBRA OOF 重新标定结果。
- 当前主线外部复核报告：
  - `artifacts/reports/busi_demo_unetplusplus_resnet34_mainline_review.json`

## 定制优化实验

### 1. 权重 / Stacker / Blend / Area Gate 局部搜索

- 输入：BUSBRA full-image OOF、U-Net++ ResNet34 predicted-mask ROI OOF、ROI area ratio。
- 搜索范围：full-image ConvNeXt/EfficientNet 权重、ROI ConvNeXt/EfficientNet 权重、二特征 stacker、ROI blend、area gate、OOF 阈值。
- 当前 U-Net++ OOF AUC：`0.924289`。
- 最佳局部候选 OOF AUC：`0.923960`。
- 结论：内部 OOF 未提升，不做 BUSI 外验，不合入。
- 报告：
  - `unetplusplus_resnet34_custom_roi_optimization.md`
  - `unetplusplus_resnet34_custom_roi_optimization.json`

### 2. Descriptor Router

- 输入：BUSBRA OOF 概率、U-Net++ ResNet34 predicted-mask descriptor、ROI 形态与边界质量特征。
- 内部 OOF：
  - baseline AUC：`0.923439`
  - router AUC：`0.927110`
  - Sensitivity：`0.8237 -> 0.8402`
  - Specificity：`0.8825 -> 0.8667`
  - F1：`0.7962 -> 0.7932`
- 冻结外部复核：
  - BUSI AUC：`0.906631`
  - Sensitivity：`0.8524`
  - Specificity：`0.7963`
  - F1：`0.7490`
- 结论：内部 OOF 提升未能迁移到 BUSI，外部泛化不足，不合入。
- 报告：
  - `descriptor_router_unetplusplus_resnet34_oof_protocol_relaxed.md`
  - `descriptor_router_unetplusplus_resnet34_oof_protocol_relaxed.json`
  - `busi_unetplusplus_resnet34_descriptor_router_relaxed_review.json`

## 验证

- 主线 BUSI 复核命令：
  - `python scripts/eval_busi.py --config configs/inference/demo.yml --output artifacts/reports/busi_demo_unetplusplus_resnet34_mainline_review.json`
- Descriptor router 冻结外验命令：
  - `python scripts/eval_busi.py --config "artifacts/reports/Chinese reports/09_unetplusplus_resnet34_custom_roi/frozen_configs/demo_unetplusplus_resnet34_descriptor_router_relaxed.yml" --output "artifacts/reports/Chinese reports/09_unetplusplus_resnet34_custom_roi/busi_unetplusplus_resnet34_descriptor_router_relaxed_review.json"`
- 单元测试：
  - `51 passed in 7.22s`
