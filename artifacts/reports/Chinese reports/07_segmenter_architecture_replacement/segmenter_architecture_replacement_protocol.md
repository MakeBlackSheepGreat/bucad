# 非 SAM 分割模型替换主线 ROI 外部验证

## 实验边界

- 训练集：BUSBRA，用于分割器训练和内部分割指标记录。
- 外部验证：BUSI，每个冻结配置只运行一次，不根据 BUSI 结果回调分割器或 ROI 参数。
- 主线分类器、分类阈值、ROI stacker、ROI area gate 保持 `configs/inference/demo.yml` 原设置；本实验只替换分割器 checkpoint。
- SAM、BUSSAM、Auto-BUSAM 等 prompt/adapter 路线未纳入本轮实验。

## 主线对照

- BUSI AUC：`0.9256`
- Threshold：`0.51`
- Accuracy / Sensitivity / Specificity / F1：`0.8532` / `0.8667` / `0.8467` / `0.7930`

## 结果总表

| 方法 | 类别 | 状态 | Val Dice | Val IoU | Boundary F1 | HD95 | BUSI AUC | BUSI Acc | Sens | Spec | F1 | 决策 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| unetplusplus_densenet121_bce_dice | U-Net++ / DenseNet121 | trained | 0.8620 | 0.7841 | 0.6997 | 17.55 | 0.9175 | 0.8377 | 0.8714 | 0.8215 | 0.7771 | report_only |
| unetplusplus_resnet34_bce_dice | U-Net++ / stronger ResNet | trained | 0.8068 | 0.7117 | 0.6245 | 24.97 | 0.9190 | 0.8501 | 0.8571 | 0.8467 | 0.7877 | report_only |
| unetplusplus_efficientnetb0_bce_dice | U-Net++ / EfficientNet | trained | 0.8849 | 0.8105 | 0.7128 | 14.73 | 0.9148 | 0.8470 | 0.8524 | 0.8444 | 0.7834 | report_only |
| fpn_efficientnetb0_bce_dice | multi-scale feature pyramid | trained | 0.8847 | 0.8064 | 0.6787 | 11.30 | 0.9164 | 0.8485 | 0.8333 | 0.8558 | 0.7812 | report_only |
| fpn_densenet121_bce_dice | multi-scale DenseNet | trained | 0.8636 | 0.7822 | 0.6648 | 14.75 | 0.9096 | 0.8408 | 0.8190 | 0.8513 | 0.7696 | report_only |
| deeplabv3plus_resnet34_bce_dice | atrous context / DeepLabV3+ | trained | 0.8811 | 0.8022 | 0.6990 | 11.95 | 0.9124 | 0.8423 | 0.8381 | 0.8444 | 0.7753 | report_only |
| manet_resnet34_bce_dice | attention decoder | trained | 0.7585 | 0.6526 | 0.5569 | 34.05 | 0.9145 | 0.8501 | 0.8095 | 0.8696 | 0.7780 | report_only |
| linknet_densenet121_bce_dice | light decoder / DenseNet | trained | 0.8763 | 0.7974 | 0.6896 | 13.68 | 0.9145 | 0.8377 | 0.8524 | 0.8307 | 0.7732 | report_only |
| pan_resnet34_bce_dice | pyramid attention | trained | 0.8606 | 0.7752 | 0.6532 | 13.58 | 0.9141 | 0.8331 | 0.8238 | 0.8375 | 0.7621 | report_only |
| pspnet_resnet34_bce_dice | pyramid pooling context | trained | 0.8432 | 0.7517 | 0.6157 | 19.65 | 0.9056 | 0.8501 | 0.8143 | 0.8673 | 0.7790 | report_only |

## 结论

- 本轮最佳替换分割器为 `unetplusplus_resnet34_bce_dice`，BUSI AUC `0.9190`，相对主线变化 `-0.0066`。
- 若 `决策` 为 `report_only`，表示该方法未达到“外部 AUC 高于主线且关键指标无明显下降”的候选保留条件。
- 本报告不自动修改 `configs/inference/demo.yml`，也不写入 README；是否合入需要人工选择。
