# Stage Review: validation (C:\\Users\\Sakura\\Desktop\\BlackSheep\\Agent)

- **Date**: 2026-08-20
- **Reviewer**: independent-subagent (neural-paper-stage-review / validation 模块) via luna_worker
- **Stage verdict**: PASS
- **审查依据**：仅 experiment-registry.csv, evidence-map.md, paper_evidence/lesionext_lea_component_ablation.json, lesionext_lens_v1a_evidence_only_5fold_oof.json

## 逐维判定

| 维度 | 判定（pass / needs-evidence / revise） | 证据锚点（文件/行号/实验编号/数字） | 说明 |
| --- | --- | --- | --- |
| 指标口径 | pass | pooled OOF 为 image-level pooled, fold-mean仅作secondary, 与 task-types.json classification_prob 一致 | 未混用聚合口径 |
| 统计方法 | pass | paired case-cluster bootstrap 10000 seed 20260811, 外部1000, CI [-0.003,0.032] 正确描述为descriptive | 统计方法参数正确且未夸大 |
| 泄漏断言 | pass | busbra_5fold_splits.csv case-level disjoint, 外部锁集未参与选择, BUS-UCLM与Small已排除 | 无泄漏 |
| 消融完整性 | pass | head-only 0.9072 vs LEA 0.9150 vs LEA+local 0.8907, 五指标全报, 权重0.10-0.30筛查已归档 | 必要性消融完整 |
| 阈值协议 | pass | 固定0.50, Youden仅诊断, 无阈值塌缩 | 协议冻结 |
| 领域分层 | pass | 内部与三外部单独报告, 未将外部当选择集 | 分层诚实 |

## 必改项
无

## 复核要求
无

## 结论

Validation阶段统计口径与泄漏控制均已过门，结果可重复且比较公平，无必改缺陷。
