# SonoGloReNet MixStyle 补充实验报告

## 实验日期
- 2026-06-19

## 目的
- 验证 MixStyle 这种轻量域泛化模块，能否改善 BUSBRA -> BUSI 的跨域泛化。

## 参考行
- `posw17`
- fold1 内部 AUC：`0.917385`
- fold1 BUSI AUC：`0.904626`

## 候选：MixStyle
- 配置：`configs/classifier/sonoglore_convnext_tiny_grn_eca_asymproj_posw17_mixstyle.yml`
- 作用位置：stage2 / stage3
- fold1 内部 AUC：`0.903000`
- fold1 内部 sensitivity：`0.811475`
- fold1 内部 specificity：`0.814229`
- fold1 BUSI AUC：`0.881601`
- fold1 BUSI sensitivity：`0.790476`
- fold1 BUSI specificity：`0.800915`
- 结论：淘汰

## 结果解读
- MixStyle 让模型更偏向恶性判断。
- 内部和外部的 sensitivity 都有抬升倾向。
- 但 specificity 明显下滑，BUSI 外部 AUC 下降较多。
- 这说明当前域移位问题，并不能仅靠训练期特征统计混合直接解决。

## 与前两条线合并后的判断
- ArcMargin / SupCon：把模型推向更保守区域。
- MixStyle：把模型推向更激进区域。
- 两种方向都没有带来内外双提升。
- 因此更像是 decision bias 在域间不稳定，而不是简单的分离度不足。

## 下一步建议
- 停止继续堆新的大 loss 或泛化模块。
- 转向：
  1. learnable logit bias correction
  2. validation-constrained threshold shaping
  3. 外部更稳的概率整形
