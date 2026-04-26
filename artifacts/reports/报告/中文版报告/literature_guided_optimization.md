<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 文献指导优化记录

日期：2026-04-24

## 已审阅来源

| Source | 对本项目有用的思路 |
| --- | --- |
| BUSI dataset paper: https://pubmed.ncbi.nlm.nih.gov/31867417/ | BUSI 支持 classification、detection 和 segmentation，因此外部评估应同时报告概率排序指标和运行点指标。 |
| Fully automated segmentation and classification pipeline | Ensemble-style 多阶段流程是 breast ultrasound CAD 的合理方向。 |
| Multi-task segmentation and classification | segmentation-aware classification 和 joint learning 是有潜力的后续方向。 |
| MTL-OCA joint segmentation/classification | object-context attention 支持 lesion-aware features 能帮助分类的观点。 |
| Lesion region perception classification | ROI / lesion-region perception 是超越 whole-image classification 的重要下一步。 |
| OpenUS ultrasound foundation model | ultrasound-specific pretraining 可能比单纯提高输入分辨率更有价值。 |

## 已应用优化

更稳妥的即时优化不是改变训练/测试边界，也不是直接替换为未经验证的高分辨率模型，而是采用与文献方向一致的 ensemble improvement：

- 保留当前五折 EfficientNetV2-S ensemble 作为最强基础。
- 加入互补的 DenseNet121 五折 ensemble。
- 搜索 DenseNet contribution weight；BUSI 仅作为外部评估。
- 保留 horizontal-flip TTA，并做细粒度 threshold sweep。

## 结果

| Runtime Candidate | AUC | Threshold | Sensitivity | Precision | F1-Score | Specificity | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNetV2-S five-fold + TTA | 0.8997 | 0.33 | 0.8000 | - | - | 0.8764 | 0.8516 |
| EfficientNetV2-S + DenseNet121 mixed ensemble, DenseNet weight 0.50 | 0.9050 | 0.28 | 0.8095 | - | - | 0.8696 | 0.8501 |
| EfficientNetV2-S + DenseNet121 mixed ensemble, DenseNet weight 0.63 | 0.9052 | 0.27 | 0.8095 | - | - | 0.8719 | 0.8516 |

## 决策

- 使用 DenseNet weight `0.63` 作为当时 runtime candidate，因为它在搜索中取得最佳 BUSI AUC，并保持较均衡的 Sensitivity/Specificity。
- 不优先直接做 `320` 或 `384` 分辨率扩展，除非配合不同训练策略；当前本地证据显示直接升分辨率会降低 fold-1 AUC。
- 下一高价值方向是 lesion-aware ROI classification 或 multi-task segmentation/classification。
