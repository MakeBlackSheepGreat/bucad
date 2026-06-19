# SonoGloReNet 判别边界增强补充实验报告

## 实验日期
- 2026-06-19

## 目的
- 针对当前 SonoGloReNet 在 BUSI 外部域上存在的良性高置信误报问题，验证两类更偏判别边界增强的方法：
  1. ArcFace/CosFace 风格的 margin-cosine 分类头
  2. Supervised Contrastive Learning 辅助损失

## 参考行
- `posw17`
- fold1 内部 AUC：`0.917385`
- fold1 BUSI AUC：`0.904626`

## 候选 1：ArcMargin Head
- 配置：`configs/classifier/sonoglore_convnext_tiny_grn_eca_asymproj_posw17_arcmargin.yml`
- fold1 内部 AUC：`0.907341`
- fold1 内部 specificity：`0.924901`
- fold1 内部 sensitivity：`0.721311`
- fold1 BUSI AUC：`0.867457`
- 结论：淘汰

## 候选 2：SupCon Auxiliary Loss
- 配置：`configs/classifier/sonoglore_convnext_tiny_grn_eca_asymproj_posw17_supcon.yml`
- fold1 内部 AUC：`0.911002`
- fold1 内部 specificity：`0.913043`
- fold1 内部 sensitivity：`0.721311`
- fold1 BUSI AUC：`0.894917`
- 结论：淘汰

## 共同现象
- 两条线都让分类器更保守。
- 内部良性 FP 的确下降，specificity 有提升。
- 但恶性召回同步下降。
- 外部 BUSI AUC 都没有守住参考线。

## 结论
- 仅靠更强的判别边界约束，并不能解决当前 SonoGloReNet 的主矛盾。
- 当前问题更像是概率偏置、域间校准漂移、以及外部域决策边界位置不稳。

## 下一步建议
- 优先转向：
  1. 后验温度校准
  2. learnable logit bias correction
  3. 面向外部域稳定性的概率整形
