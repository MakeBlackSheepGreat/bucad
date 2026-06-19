# SonoGloReNet 消融矩阵 fold1 筛选报告

> 数据边界说明：BUSBRA 仅用于训练、内部验证和模型选择；BUSI 仅用于锁定外部评估。本报告复用现有 `B0 / R0 / R1` 参考行，不重复训练。

日期：2026-06-19

## 固定参考行

| 编号 | 角色 | 内部验证 AUC | BUSI AUC | BUSI Balanced Accuracy | Sensitivity | Specificity | F1-Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | baseline_reference | 0.9259 | 0.8926 | 0.8229 | 0.7762 | 0.8696 | 0.7581 |
| R0 | reference_sonoglore | 0.9096 | 0.8782 | 0.8006 | 0.8048 | 0.7963 | 0.7222 |
| R1 | appendix_failed_relu | 0.9068 | 0.5711 | 0.5173 | 0.0667 | 0.9680 | 0.1176 |

## 第一阶段：结构消融排序

| Rank | Candidate | Internal AUC | BUSI AUC | BUSI Balanced Accuracy | Sensitivity | Specificity | F1-Score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | S1 / ms_noattn | 0.9106 | 0.8809 | 0.8175 | 0.7952 | 0.8398 | 0.7472 |
| 2 | S2 / stage4_only | 0.9081 | 0.8827 | 0.8107 | 0.7952 | 0.8261 | 0.7373 |
| 3 | S3 / stage34_noattn | 0.9102 | 0.8754 | 0.7874 | 0.7762 | 0.7986 | 0.7072 |

## 第二阶段：正则矩阵排序

| Rank | Candidate | Internal AUC | BUSI AUC | BUSI Balanced Accuracy | Sensitivity | Specificity | F1-Score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | S1_T0 / ms_noattn__base | 0.9106 | 0.8809 | 0.8175 | 0.7952 | 0.8398 | 0.7472 |
| 2 | S1_T3 / ms_noattn__regularized_cutmix | 0.9181 | 0.8853 | 0.8117 | 0.7905 | 0.8330 | 0.7394 |
| 3 | S1_T1 / ms_noattn__regularized | 0.9151 | 0.8709 | 0.7852 | 0.7238 | 0.8467 | 0.7086 |
| 4 | S1_T2 / ms_noattn__cutmix | 0.9119 | 0.9018 | 0.7483 | 0.9429 | 0.5538 | 0.6567 |

## 晋级门控

- 参考 `R0` 的 BUSI balanced accuracy：`0.8006`
- 参考 `R0` 的最低允许 AUC：`0.8732`
- 当前候选：`S1_T0` / `ms_noattn__base`
- 当前候选 balanced accuracy：`0.8175`
- 当前候选 AUC：`0.8809`
- 门控结果：`晋级 5-fold`

## 最终候选概览

- 候选：`S1_T0` / `ms_noattn__base`
- 配置：`C:\Users\876762~1\AppData\Local\Temp\sonoglore_ablation_matrix\ms_noattn__base_fold1.yml`
- 推理配置：`C:\Users\876762~1\AppData\Local\Temp\sonoglore_ablation_matrix\ms_noattn__base_fold1_inference.yml`
- 内部验证 AUC：`0.9106`
- BUSI AUC：`0.8809`
- BUSI balanced accuracy：`0.8175`

## 错判与概率分布

| 候选 | newly_worse | newly_better | both_wrong |
| --- | ---: | ---: | ---: |
| S1 / ms_noattn | 47 | 38 | 66 |
| S2 / stage4_only | 58 | 43 | 61 |
| S3 / stage34_noattn | 63 | 32 | 72 |
| S1_T0 / ms_noattn__base | 47 | 38 | 66 |
| S1_T1 / ms_noattn__regularized | 47 | 26 | 78 |
| S1_T2 / ms_noattn__cutmix | 138 | 35 | 69 |
| S1_T3 / ms_noattn__regularized_cutmix | 46 | 33 | 71 |

- baseline 正确 -> 当前候选错误：`47`
- baseline 错误 -> 当前候选正确：`38`
- 双方都错：`66`
- benign 恶性概率均值：baseline `0.1412`，当前 `0.1920`
- malignant 恶性概率均值：baseline `0.7627`，当前 `0.7781`
- 差异 CSV：`C:\Users\876762330\Desktop\projects\Agent\artifacts\reports\generated\ms_noattn__base_vs_baseline_case_diff.csv`
- 新增误判 CSV：`C:\Users\876762330\Desktop\projects\Agent\artifacts\reports\generated\ms_noattn__base_worsened_cases.csv`

### 代表性新增误判 Top-N

- `benign (366)` / `benign`: baseline `0.0035` -> current `0.9845` (shift `0.9810`)
- `benign (360)` / `benign`: baseline `0.0132` -> current `0.9597` (shift `0.9465`)
- `benign (285)` / `benign`: baseline `0.0001` -> current `0.9328` (shift `0.9327`)
- `benign (63)` / `benign`: baseline `0.0170` -> current `0.9259` (shift `0.9089`)
- `malignant (147)` / `malignant`: baseline `0.9668` -> current `0.0687` (shift `-0.8981`)
- `benign (162)` / `benign`: baseline `0.0726` -> current `0.9707` (shift `0.8981`)
- `benign (284)` / `benign`: baseline `0.0003` -> current `0.8769` (shift `0.8767`)
- `benign (234)` / `benign`: baseline `0.0095` -> current `0.8790` (shift `0.8694`)
- `benign (159)` / `benign`: baseline `0.1069` -> current `0.9291` (shift `0.8223`)
- `malignant (161)` / `malignant`: baseline `0.9748` -> current `0.1545` (shift `-0.8204`)

## 单图推理耗时

- 测试样本：`benign (1)`
- 预热次数：`2`，统计次数：`10`
- `B0 baseline`: mean `0.01572` s, min `0.01475` s, max `0.01648` s
- `Final candidate`: mean `0.01796` s, min `0.01558` s, max `0.02093` s

## 结论

- 本轮排序固定按 BUSI balanced accuracy、BUSI AUC、BUSI F1、内部验证 AUC 依次判定。
- 当前最终候选门控结果：`已进入 5-fold`。
- `R1` 仅保留在附录语境下作为失败参考，不参与晋级判断。
