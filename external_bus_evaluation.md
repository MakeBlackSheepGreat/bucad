# 独立乳腺超声外部评估

## 数据集队列

| 队列 | 公开来源 | 适合任务 | 当前状态 |
| --- | --- | --- | --- |
| BUSI | Al-Dhabyani 等公开 BUSI 集 | 分类、分割 | 已有复核入口 |
| BUS-UCLM | Mendeley Data，DOI `10.17632/7fvgj4jsp7.1` | 分类、分割 | 已下载并完成首次冻结评估 |
| BUSI-WHU | Mendeley Data，DOI `10.17632/k6cpmwybk3.1` | 分类、分割 | 已完成 SHA-256 标签映射并完成首次冻结评估 |
| TCIA BrEaST / BREAST-LESIONS-USG | The Cancer Imaging Archive | 分类、分割 | 已完成 XLSX 病理标签对齐并完成首次冻结评估 |
| UDIAT Dataset B | UDIAT 公开研究数据 | 分割、少量分类 | 等待许可和标签核验 |
| OASBUD | Zenodo | 原始 RF / 域偏移 | 需要 B-mode 重建，不直接并入图像评估 |

## BUS-UCLM 首次结果

- 队列：264 张带病理标签图像，174 benign、90 malignant；normal 样本按预先定义的二分类协议排除。
- 模型：冻结 LesioNeXt-MoE v3 五折 crop-sweep 配置。
- 阈值：使用训练配置内固定阈值 `0.50`，没有在 BUS-UCLM 上调阈值、选择 checkpoint 或选择 TTA。
- AUC：`0.9109`，bootstrap 95% CI `[0.8703, 0.9425]`。
- Sensitivity：`0.8333`；Specificity：`0.8161`；Accuracy：`0.8220`；F1：`0.7614`。
- 分割：264 个样本都有 RGB mask，Dice `0.5895`，IoU `0.4771`。

原始 JSON、逐样本预测 CSV 均写入 `artifacts/reports/`。这些报告被 `.gitignore` 忽略，原始数据位于 `data/external/`，不进入 Git。

## 新增队列结果

### BUSI-WHU

- 788 例可由原始图像和标签参考通过 SHA-256 对齐，476 benign、312 malignant。
- AUC：`0.7966`，bootstrap 95% CI `[0.7634, 0.8279]`；Sensitivity：`0.6859`；Specificity：`0.7521`；Accuracy：`0.7259`；F1：`0.6646`；Dice：`0.7748`；IoU：`0.6693`。

### TCIA BrEaST

- 临床表共 256 例，其中 252 例属于 benign/malignant 二分类协议，154 benign、98 malignant；每例都有肿瘤 mask。
- AUC：`0.8292`，bootstrap 95% CI `[0.7731, 0.8776]`；Sensitivity：`0.8265`；Specificity：`0.7078`；Accuracy：`0.7540`；F1：`0.7232`；Dice：`0.6939`；IoU：`0.5835`。

## 冻结评估规则

1. BUSBRA 只用于训练和内部 OOF；BUSI、BUS-UCLM、BUSI-WHU、TCIA、UDIAT 只用于外部测试。
2. 外部队列不参与阈值、模型、TTA、预处理和 checkpoint 选择。
3. 每个队列单独输出 AUC、Sensitivity、Specificity、Accuracy、Precision、F1、bootstrap CI；有 mask 时追加 Dice、IoU。
4. 病人级重复图像必须用 `case_id` 去重，正常类只在任务协议允许时纳入。
5. “SOTA”只在任务定义、标签协议、数据划分和指标完全一致时使用；多队列泛化结果单独报告，避免跨协议排名。

## 运行方式

```powershell
conda run -n BUCAD python scripts/list_external_datasets.py --root data/external
conda run -n BUCAD python scripts/eval_external_bus.py `
  --dataset bus_uclm `
  --root data/external/bus_uclm `
  --manifest data/external/bus_uclm/manifest.csv `
  --config configs/inference/lesionext_moe_v3_5fold_tta_crop_sweep.yml `
  --output artifacts/reports/external_bus_uclm_lesionext_v3.json
```
