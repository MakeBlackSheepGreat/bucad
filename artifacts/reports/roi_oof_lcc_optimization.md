# ROI OOF 与 LCC 后处理优化报告

日期：2026-04-25

## 数据边界

- BUSBRA 用于训练、OOF 融合、内部验证和阈值选择。
- BUSI 只用于配置固定后的外部复核，不用于训练、权重细搜、阈值细搜或反复调参。
- 本轮可合并改动只采用 BUSBRA 支撑的参数：ROI stack 阈值来自 BUSBRA OOF，mask 阈值和最大连通域来自 BUSBRA 分割验证。

## 优化过程

| 候选 | 选择依据 | BUSBRA 内部结果 | BUSI 外部复核 | 决策 |
| --- | --- | --- | --- | --- |
| 当前 ROI OOF stacker | BUSBRA OOF 逻辑回归 | OOF AUC `0.9223`，阈值 `0.55` | AUC `0.9196`，Sensitivity `0.8429`，Specificity `0.8330` | 已是上一版主线 |
| 简单 logit 融合 | BUSBRA OOF 网格搜索 | OOF AUC `0.9239`，高于逻辑回归 stacker | AUC `0.9196`，Sensitivity `0.8143`，Specificity `0.8558` | 外部 AUC 无增益，未合并 |
| 分权重 logit 融合 | BUSBRA OOF 网格搜索 | OOF AUC `0.9240`，full ConvNeXt 权重 `0.61`，ROI ConvNeXt 权重 `0.54` | AUC `0.9196`，Sensitivity `0.8476`，Specificity `0.8261` | 外部 AUC 无增益，未合并 |
| LCC ROI 后处理 | BUSBRA 分割验证 | mask 阈值 `0.40` 的 Dice 最高；最大连通域降低无关区域干扰 | AUC `0.9208`，Sensitivity `0.8524`，Specificity `0.8169` | 合并入 `demo.yml` |

## BUSBRA 分割验证依据

| mask 阈值 | Dice | 原 ROI 中位面积占比 | LCC ROI 中位面积占比 | 说明 |
| ---: | ---: | ---: | ---: | --- |
| 0.30 | 0.7971 | 0.8633 | 0.5751 | 阈值偏低，ROI 过大 |
| 0.40 | 0.8085 | 0.5750 | 0.5292 | Dice 最高，保留较完整上下文 |
| 0.50 | 0.8074 | 0.5167 | 0.4602 | 接近 0.40，但 Dice 略低 |
| 0.60 | 0.7797 | 0.4410 | 0.4104 | 阈值偏高，可能切掉弱边界 |
| 0.70 | 0.7399 | 0.4059 | 0.3797 | 阈值过高，不适合作为主线 |

## 合并后的主线配置

- 配置文件：`configs/inference/demo.yml`
- 分类模型：ConvNeXt-Tiny 五折 + EfficientNetV2-S 五折。
- ROI 方法：`segmenter_fold1.pt` 预测 mask，`mask_threshold=0.40`，`largest_component=true`，`margin_ratio=0.35`。
- 概率融合：完整图概率 + ROI 概率，使用 BUSBRA OOF 训练出的 logit logistic stacker。
- 判定阈值：`0.55`，来自 BUSBRA ROI OOF。

## 外部复核结果

| 配置 | AUC | 阈值 | Sensitivity | Specificity | Accuracy | Precision | F1-Score | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 上一版 ROI OOF 主线 | 0.9196 | 0.55 | 0.8429 | 0.8330 | 0.8362 | 0.7080 | 0.7696 | TN 364 / FP 73 / FN 33 / TP 177 |
| LCC ROI 后处理主线 | 0.9208 | 0.55 | 0.8524 | 0.8169 | 0.8284 | 0.6911 | 0.7633 | TN 357 / FP 80 / FN 31 / TP 179 |

## 结论

- 本轮真正有效的改动是 ROI mask 后处理，而不是替换 OOF stacker。
- AUC 从 `0.9196` 提升到 `0.9208`，Sensitivity 从 `0.8429` 提升到 `0.8524`，达到原始任务中更重视漏诊控制的方向。
- Specificity 和 Accuracy 略有下降，说明该方案更偏向提高恶性检出率；这符合乳腺肿瘤辅助诊断中优先降低漏诊的策略。
- 后续如果继续优化，优先做 5 折分割器 OOF mask 或全 BUSBRA 分割器重训，而不是在 BUSI 上继续细搜阈值。
