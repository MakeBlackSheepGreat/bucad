<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 答辩问答准备

日期：2026-04-24

## 为什么要分离 BUSBRA 和 BUSI？

BUSBRA 用于训练和内部验证，BUSI 保留给外部评估和 demo 验证。这样可以避免测试集影响模型训练，使最终评估更可信。

## 如何防止数据泄漏？

项目使用病例级 split，而不是随机图像级 split。生成的 split summary 记录了 1,064 个唯一病例，并且 `leakage_detected=false`。

## 为什么使用 EfficientNetV2-S？

开发手册选择 EfficientNetV2-S 作为最终候选，是因为它属于现代、参数效率较高的 CNN 家族，适合有限医学影像数据。当前仓库已经准备好配置，但最终优势结论应等 T059/T060 正式训练和对比完成后再表述。

## Grad-CAM 能证明什么？

Grad-CAM 不能证明临床正确性。它提供的是模型关注区域的可视解释，帮助评审和队员识别明显不匹配或不可靠的行为。

## 当前指标有哪些局限？

当时 BUSI 最终评估在阈值 0.50 下达到 AUC `0.7564`、Sensitivity `0.6095`、Specificity `0.8009`、Accuracy `0.7388`。Sensitivity 仍低于目标，因此系统应定位为 prototype。

## 它能替代医生吗？

不能。UI 和答辩都应声明：系统仅用于辅助分析、教学和比赛演示，不能替代医生判断。

## 分割或解释失败时会怎样？

推理契约支持 graceful degradation：分类仍可返回，同时界面会用 warning 展示可视化缺失原因。
## 比赛指标完整性说明

比赛统一指标集为 AUC、Accuracy、Recall/Sensitivity、Precision、Specificity 和 F1-Score。如果历史归档表中 Precision 或 F1-Score 显示 `-`，说明原始摘要未保留可反推该值的 confusion matrix 或原始概率。锁定 BUSI 运行点结果以 `artifacts/reports/competition_metrics_all_busi_reports.md` 作为完整指标来源。

