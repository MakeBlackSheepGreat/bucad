<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 可视化证据复核

日期：2026-04-24

## 已复核输出

| Case | Original | Lesion Overlay | Grad-CAM | Review Status | Notes |
| --- | --- | --- | --- | --- | --- |
| `001_benign_(1)` | present | present | present | pass for demo evidence | Overlay 和 heatmap 均可用；临床合理性仍需队伍人工复核。 |
| `002_benign_(10)` | present | present | present | pass for demo evidence | Overlay 和 heatmap 均可用；临床合理性仍需队伍人工复核。 |
| `003_benign_(100)` | present | present | present | pass for demo evidence | Overlay 和 heatmap 均可用；临床合理性仍需队伍人工复核。 |
| `004_benign_(101)` | present | present | present | pass for demo evidence | Overlay 和 heatmap 均可用；临床合理性仍需队伍人工复核。 |
| `005_benign_(102)` | present | present | present | pass for demo evidence | Overlay 和 heatmap 均可用；临床合理性仍需队伍人工复核。 |
| `006_benign_(103)` | present | present | present | pass for demo evidence | Overlay 和 heatmap 均可用；临床合理性仍需队伍人工复核。 |

## 复核总结

- 所有导出的最终可视化样例都存在 segmentation overlay 文件。
- 所有导出的最终可视化样例都存在 Grad-CAM explanation 文件。
- 最终 visual export 命令中未观察到 missing-output reason。
- 局限：导出样例放入最终 PPT/报告前，仍应人工筛查其临床合理性。
