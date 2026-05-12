# U-Net++ ResNet34 主线定制 ROI 标定实验

## 数据边界

- 本实验只读取 BUSBRA OOF 分类概率、U-Net++ ResNet34 预测 mask 的 ROI OOF 概率与 ROI 面积缓存。
- BUSI 不参与本脚本中的权重、stacker、area gate、blend 或阈值选择。
- 输出的 frozen config 是冻结候选，后续只能做一次外部复核，不能根据外验结果反向调参。

## 当前 U-Net++ 配置 OOF

- AUC `0.924289`，threshold `0.60`，Sensitivity `0.8237`，Specificity `0.8793`，F1 `0.7937`。

## 入选定制配置

- AUC `0.923960`，threshold `0.39`，Sensitivity `0.8435`，Specificity `0.8517`，F1 `0.7835`。
- Full-image ConvNeXt 权重：`0.600`；ROI ConvNeXt 权重：`0.573`。
- Stacker：`logit` 特征，`C=1.0`，`class_weight=None`，CV AUC `0.923111`。
- ROI blend：`0.75`；area gate：`0.08-1.01`，fallback `24` 个样本。
- 冻结配置：`artifacts\reports\Chinese reports\09_unetplusplus_resnet34_custom_roi\frozen_configs\demo_unetplusplus_resnet34_custom_roi.yml`。

## Top 10 候选

| Rank | OOF AUC | Threshold | Sens | Spec | F1 | Full Conv | ROI Conv | Blend | Gate | Stacker |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.923960 | 0.39 | 0.8435 | 0.8517 | 0.7835 | 0.600 | 0.573 | 0.75 | 0.08-1.01 | logit C=1.0 None |
| 2 | 0.923947 | 0.38 | 0.8484 | 0.8462 | 0.7821 | 0.600 | 0.600 | 0.75 | 0.08-1.01 | logit C=1.0 None |
| 3 | 0.923943 | 0.39 | 0.8451 | 0.8509 | 0.7838 | 0.600 | 0.625 | 0.75 | 0.08-1.01 | logit C=1.0 None |
| 4 | 0.923910 | 0.41 | 0.8418 | 0.8557 | 0.7855 | 0.600 | 0.600 | 0.85 | 0.08-1.01 | logit C=1.0 None |
| 5 | 0.923889 | 0.41 | 0.8451 | 0.8557 | 0.7874 | 0.600 | 0.573 | 0.85 | 0.08-1.01 | logit C=1.0 None |
| 6 | 0.923887 | 0.39 | 0.8501 | 0.8470 | 0.7836 | 0.600 | 0.625 | 0.85 | 0.08-1.01 | logit C=1.0 None |
| 7 | 0.923865 | 0.39 | 0.8435 | 0.8517 | 0.7835 | 0.573 | 0.625 | 0.75 | 0.08-1.01 | logit C=1.0 None |
| 8 | 0.923832 | 0.44 | 0.8204 | 0.8754 | 0.7886 | 0.573 | 0.600 | 0.75 | 0.08-1.01 | logit C=1.0 None |
| 9 | 0.923806 | 0.42 | 0.8353 | 0.8612 | 0.7860 | 0.573 | 0.625 | 0.85 | 0.08-1.01 | logit C=1.0 None |
| 10 | 0.923795 | 0.42 | 0.8353 | 0.8620 | 0.7867 | 0.573 | 0.600 | 0.85 | 0.08-1.01 | logit C=1.0 None |

## 结论

- 内部 OOF AUC 增量：`-0.000329`。
- 建议：`report_only_no_internal_oof_gain`。
