# 分割器替换后的 ROI OOF 重新标定实验

## 协议边界

- BUSBRA 用于分割器训练、预测 mask OOF、ROI stacker 拟合、area gate 扫描和阈值选择。
- BUSI 只在每个配置冻结后做一次外部验证，不根据 BUSI 回调参数。
- 每个方法使用 fold-specific segmenter 生成 BUSBRA OOF mask；外部部署配置使用 5 个 segmenter checkpoint 的 mask ensemble。
- 主线对照为当前 `configs/inference/demo.yml`：BUSI AUC `0.9256`，阈值 `0.51`。

## 总表

| 方法 | 状态 | Val Dice | OOF ROI AUC | OOF Stack AUC | Gate | OOF Gated AUC | BUSI AUC | Sens | Spec | F1 | 决策 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| pan_resnet34_bce_dice | trained_recalibrated | 0.8547 | 0.8771 | 0.9232 | 0.05-1.01 | 0.9232 | 0.9223 | 0.8810 | 0.8078 | 0.7724 | report_only |
| fpn_densenet121_bce_dice | trained_recalibrated | 0.8497 | 0.8737 | 0.9235 | 0.03-1.01 | 0.9237 | 0.9217 | 0.8952 | 0.8009 | 0.7753 | report_only |
| unetplusplus_densenet121_bce_dice | trained_recalibrated | 0.8742 | 0.8869 | 0.9251 | 0.00-1.01 | 0.9251 | 0.9227 | 0.8238 | 0.8581 | 0.7775 | report_only |
| manet_resnet34_bce_dice | trained_recalibrated | 0.8058 | 0.8718 | 0.9229 | 0.08-1.01 | 0.9230 | 0.9232 | 0.8143 | 0.8650 | 0.7773 | report_only |
| linknet_densenet121_bce_dice | trained_recalibrated | 0.8840 | 0.8782 | 0.9225 | 0.00-1.01 | 0.9225 | 0.9212 | 0.8286 | 0.8604 | 0.7820 | report_only |
| pspnet_resnet34_bce_dice | trained_recalibrated | 0.8538 | 0.8804 | 0.9243 | 0.03-1.01 | 0.9245 | 0.9217 | 0.8381 | 0.8490 | 0.7788 | report_only |

## 结论

- 本轮重新标定后最佳方法为 `manet_resnet34_bce_dice`，BUSI AUC `0.9232`，相对主线 `-0.0024`。
- 只有 `candidate_hold_for_user` 才表示可供人工决定是否合入；`report_only` 只保留实验记录。
- 本脚本不修改 `configs/inference/demo.yml`，所有冻结候选配置均保存在本目录 `frozen_configs/` 下。
