# SonoGloReNet OOF-FN 重加权补充实验报告

## 实验日期
- 2026-06-19

## 目的
- 验证更保守的恶性 FN 重加权，能否在不进一步放大良性 FP 的前提下，同时改善 SonoGloReNet 的内部与外部表现。

## 先修正的实验设计问题
- 先前的 `fold1` 权重 CSV 来自 `fold1` 验证集预测，和 `fold1 train` 没有样本交集。
- 这会导致训练配置虽然加载了权重文件，但实际没有任何训练样本被加权。
- 本轮已修正为基于 `5-fold OOF` 的全量 BUSBRA 权重表。
- 同时在训练侧新增 `sample_weight_hit_count` 统计与 0-hit 保护，后续若权重文件没有命中训练样本，会直接报错。

## 参考行
- `posw17`：当前外部最强的 SonoGloReNet 单模型参考线。
- fold1 内部 AUC：`0.917385`
- fold1 BUSI AUC：`0.904626`
- 5-fold OOF AUC：`0.894437`

## 候选 1：fnonly_mild
- 权重文件：`artifacts/reports/generated/sonoglore_posw17_oof_sample_weights_fnonly_mild.csv`
- 全量加权样本数：`133`
- fold1 训练集命中数：`109`
- fold1 内部 AUC：`0.908281`，较参考 `-0.009104`
- fold1 BUSI AUC：`0.898916`，较参考 `-0.005710`
- 结论：淘汰

## 候选 2：fnextreme
- 权重文件：`artifacts/reports/generated/sonoglore_posw17_oof_sample_weights_fnextreme.csv`
- 全量加权样本数：`100`
- fold1 训练集命中数：`84`
- fold1 内部 AUC：`0.905851`，较参考 `-0.011534`
- fold1 BUSI AUC：`0.888215`，较参考 `-0.016411`
- 结论：淘汰

## 结果解读
- 这条线已经完成了“真实生效”的重加权验证。
- 更保守的恶性 FN 重加权没有带来内部提升。
- 外部 BUSI AUC 也没有守住 `posw17` 参考线。
- 说明当前主矛盾并不只是恶性漏判样本权重不足。
- 更可能的问题仍然是良性高置信误报、概率校准偏移、以及跨域泛化时的决策边界形状。

## 是否继续 OOF / 5-fold
- 不继续。
- 原因：两个候选在 fold1 BUSI AUC 都没有达到继续扩展的门槛。

## 下一步建议
- 关闭 FN-only 重加权方向。
- 优先考虑：
  1. 概率校准与对数it后处理
  2. 面向良性 FP 的抑制策略
  3. 更稳的外部域泛化约束，而不是继续堆恶性 hard mining
