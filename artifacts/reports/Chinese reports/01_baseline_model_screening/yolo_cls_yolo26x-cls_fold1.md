# YOLO Fold1 单模型对比实验

## 实验边界

- 本实验为旁路模型筛选，不修改 `configs/inference/demo.yml`。
- 训练仅使用 BUSBRA fold1 train；BUSBRA fold1 val 用于内部复核。
- BUSI 只做当前冻结单模型的一次外部验证，不用于阈值回调或二次调参。
- 本次只测试一个 YOLO 分类候选模型。

## 模型选择依据

- YOLO 候选：`yolo26x-cls.pt`。
- 选择依据：Ultralytics 官方 YOLO 分类模型表中，x 规模分类模型是同一 YOLO26 分类族内 Top-1 精度最高的公开权重。
- 任务适配：按图像分类方式训练 benign/malignant 二分类，不使用检测框或分割标签。

## 训练设置

- fold：`1`
- epochs：`30`
- imgsz：`224`
- batch：`4`
- device：`0`
- 训练样本：`1500`
- BUSBRA val 样本：`375`
- BUSI 样本：`647`
- best checkpoint：`C:\Users\876762330\Desktop\projects\Agent\artifacts\yolo_runs\yolo26x_cls_fold1\weights\best.pt`

## 结果

| 数据集 | 阈值 | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BUSBRA fold1 val | 0.50 | 0.8216 | 0.7973 | 0.6230 | 0.8814 | 0.7170 | 0.6667 | TN 223 / FP 30 / FN 46 / TP 76 |
| BUSI external | 0.50 | 0.8189 | 0.7975 | 0.6571 | 0.8650 | 0.7005 | 0.6781 | TN 378 / FP 59 / FN 72 / TP 138 |

## Youden 最优点（仅报告，不回调参数）

| 数据集 | 阈值 | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Youden J | Confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BUSBRA fold1 val | 0.50 | 0.7973 | 0.6230 | 0.8814 | 0.7170 | 0.6667 | 0.5044 | TN 223 / FP 30 / FN 46 / TP 76 |
| BUSI external | 0.49 | 0.7975 | 0.6619 | 0.8627 | 0.6985 | 0.6797 | 0.5246 | TN 377 / FP 60 / FN 71 / TP 139 |

## Fold1 参考基线（BUSI 外部，默认阈值 0.50）

| 模型 | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Small timm recipe fold1 | 0.8947 | 0.8284 | 0.7667 | 0.8581 | 0.7220 | 0.7436 |
| ConvNeXt-Tiny timm recipe fold1 | 0.8943 | 0.8423 | 0.7762 | 0.8741 | 0.7477 | 0.7617 |
| DenseNet121 fold1 | 0.8766 | 0.8083 | 0.4571 | 0.9771 | 0.9057 | 0.6076 |
| Swin-Tiny timm recipe fold1 | 0.8729 | 0.8300 | 0.7048 | 0.8902 | 0.7551 | 0.7291 |
| EfficientNetV2-S fold1 | 0.8609 | 0.7465 | 0.8333 | 0.7048 | 0.5757 | 0.6809 |
| yolo26x-cls fold1 | 0.8189 | 0.7975 | 0.6571 | 0.8650 | 0.7005 | 0.6781 |

## 初步结论

- YOLO fold1 未超过既有 ConvNeXt 单折基线，暂不建议进入主线，仅保留为旁路实验记录。
- 该结果只代表单折模型基础潜力，不代表五折集成或 ROI-aware 主线性能。
