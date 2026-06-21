# SonoGloRe-ConvNeXt V1 冻结协议

日期：2026-06-19

## 目标边界

本分支目标是设计并评估一个比 `ConvNeXt-Tiny` 更先进的现代 CNN 单模型，而不是追求 demo 主线 ensemble 的最高 AUC。

## V1 冻结结构

V1 固定为 `sonoglore_convnext_v1`：

- Backbone：ImageNet 预训练 `convnext_tiny`
- 多尺度 stage：`[2, 3, 4]`
- stage 投影维度：`[192, 256, 320]`
- stage 固定融合权重：`[0.8, 1.0, 1.0]`
- projection block：`1x1 projection + GELU + GRN + ECA`
- stage4 attention：关闭
- scale gate：关闭
- head：LayerNorm + MLP binary classifier

## 固定配置

- BUSBRA 训练配置：`configs/classifier/sonoglore_convnext_v1.yml`
- V1 新 checkpoint 推理配置：`configs/inference/sonoglore_convnext_v1_5fold_tta_crop_sweep.yml`
- 复用已训练 posw17 checkpoint 的冻结评估配置：`configs/inference/sonoglore_convnext_v1_existing_5fold_tta_crop_sweep.yml`
- 公平 ConvNeXt 对照：`configs/classifier/convnext_tiny_timm_recipe.yml`
- 公平 ConvNeXt 推理对照：`configs/inference/convnext_tiny_timm_recipe_5fold_tta_crop_sweep.yml`

## BUSBRA/BUSI 公平协议

- ImageNet 预训练：开启
- timm data config：开启
- CLAHE：开启
- 输入尺寸：`224`
- fold：`5-fold`
- TTA：`crop_pct = 0.90 / 0.95 / 1.00`，每个 crop 配 identity + hflip
- 模型比较口径：单模型 5fold ensemble，不使用 demo ROI stacker、segmenter、EfficientNet 或其他模型成员

## 已有外部结果

复用 `sonoglore_convnext_tiny_grn_eca_asymproj_posw17_5fold_tta_crop_sweep` 的已验证结果：

| 模型 | BUSI AUC | Sensitivity | Specificity | Precision | F1 | 阈值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny 5fold crop-sweep | 0.9054 | 0.8048 | 0.8719 | 0.7511 | 0.7770 | 0.46 |
| SonoGloRe-ConvNeXt V1 5fold crop-sweep | 0.9071 | 0.8190 | 0.8719 | 0.7544 | 0.7854 | 0.34 |

## ImageNet 公平性测试

ImageNet 实验不使用 BUSI/BUSBRA 的医学数据边界。它用于回答架构问题：

1. 在标准自然图像分类上，V1 是否仍保持 ConvNeXt 级别的通用表达能力。
2. V1 的 GRN/ECA/asymmetric projection 等模块分别贡献多少。
3. 参数量、吞吐、Top-1/Top-5 与 ConvNeXt-Tiny 的差异是否合理。

ImageNet-1K 原始数据需要用户通过官方 ImageNet/ILSVRC 访问或其他已授权渠道获得；仓库只提供准备、校验、训练和评估脚本。
