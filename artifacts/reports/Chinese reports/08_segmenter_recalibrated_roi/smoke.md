# 分割器替换后的 ROI OOF 重新标定实验

## 协议边界

- BUSBRA 用于分割器训练、预测 mask OOF、ROI stacker 拟合、area gate 扫描和阈值选择。
- BUSI 只在每个配置冻结后做一次外部验证，不根据 BUSI 回调参数。
- 每个方法使用 fold-specific segmenter 生成 BUSBRA OOF mask；外部部署配置使用 5 个 segmenter checkpoint 的 mask ensemble。
- 主线对照为当前 `configs/inference/demo.yml`：BUSI AUC `0.9256`，阈值 `0.51`。

## 总表

| 方法 | 状态 | Val Dice | OOF ROI AUC | OOF Stack AUC | Gate | OOF Gated AUC | BUSI AUC | Sens | Spec | F1 | 决策 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| unetplusplus_resnet34_bce_dice | failed | 0.8420 | - | - | - | - | - | - | - | - | report_only |

## 结论

- 只有 `candidate_hold_for_user` 才表示可供人工决定是否合入；`report_only` 只保留实验记录。
- 本脚本不修改 `configs/inference/demo.yml`，所有冻结候选配置均保存在本目录 `frozen_configs/` 下。
