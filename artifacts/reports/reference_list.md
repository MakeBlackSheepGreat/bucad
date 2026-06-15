# BUCAD 参考文献整理

整理日期：2026-06-12

用途：设计报告、答辩 PPT、README 技术依据、实验方法说明。本文献表按本项目当前主线配置整理：BUSBRA 训练/内部验证，BUSI 外部验证，ConvNeXt-Tiny + EfficientNetV2-S 分类集成，UNet-ResNet18 ROI 分割，Grad-CAM 可解释性，Gradio/PyInstaller 演示交付。

## 引用策略

- 数据集条目必须引用：[1] BUS-BRA、[2] BUSI。
- 方法主体优先引用：[3] ROC/AUC、[4] ConvNeXt、[5] EfficientNetV2、[6] U-Net、[7] ResNet、[8] Grad-CAM、[9] scikit-learn、[10] PyTorch。
- 工程依赖在正式报告中可按篇幅取舍；若单独说明软件实现或开源依赖，建议引用 [11]-[17]。
- 候选模型对比章节可补充引用 [18]-[21]。

## 一、数据集与评价指标

[1] GOMEZ-FLORES W, GREGORIO-CALAS M J, COELHO DE ALBUQUERQUE PEREIRA W. BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis systems[J]. Medical Physics, 2024, 51(4): 3110-3123. DOI: 10.1002/mp.16812.

适用位置：数据来源、BUSBRA 训练集、病例级划分、内部验证说明。备注：本地 `训练集/BUSBRA/LICENSE.txt` 要求使用该数据集的文档和论文引用此文献。

[2] AL-DHABYANI W, GOMAA M, KHALED H, FAHMY A. Dataset of breast ultrasound images[J]. Data in Brief, 2020, 28: 104863. DOI: 10.1016/j.dib.2019.104863. PMID: 31867417.

适用位置：BUSI 外部测试集、良性/恶性/正常三类数据说明、外部验证说明。备注：本项目只使用其中 benign 与 malignant 数据进行外部复核。

[3] FAWCETT T. An introduction to ROC analysis[J]. Pattern Recognition Letters, 2006, 27(8): 861-874. DOI: 10.1016/j.patrec.2005.10.010.

适用位置：AUC、ROC、阈值扫描、Youden J 之外的排序能力解释。

## 二、核心模型与算法方法

[4] LIU Z, MAO H, WU C Y, FEICHTENHOFER C, DARRELL T, XIE S. A ConvNet for the 2020s[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 11966-11976. DOI: 10.1109/CVPR52688.2022.01167.

适用位置：ConvNeXt-Tiny 主分类分支、timm-aware 预处理、crop-sweep TTA 的骨干网络依据。

[5] TAN M, LE Q V. EfficientNetV2: Smaller Models and Faster Training[C]//Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021, 139: 10096-10106.

适用位置：EfficientNetV2-S 辅助分类分支、CNN 家族互补集成说明。链接：https://proceedings.mlr.press/v139/tan21a.html

[6] RONNEBERGER O, FISCHER P, BROX T. U-Net: Convolutional Networks for Biomedical Image Segmentation[C]//Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015. Lecture Notes in Computer Science, 2015, 9351: 234-241. DOI: 10.1007/978-3-319-24574-4_28.

适用位置：UNet-ResNet18 分割分支、病灶 mask、ROI 裁剪、分割可视化。

[7] HE K, ZHANG X, REN S, SUN J. Deep Residual Learning for Image Recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778. DOI: 10.1109/CVPR.2016.90.

适用位置：ResNet18 编码器、UNet-ResNet18 分割器、候选 ResNet 分类基线。

[8] SELVARAJU R R, COGSWELL M, DAS A, VEDANTAM R, PARIKH D, BATRA D. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization[J]. International Journal of Computer Vision, 2020, 128(2): 336-359. DOI: 10.1007/s11263-019-01228-7.

适用位置：Grad-CAM 热力图、可解释性输出、错误样本复核。

[9] PEDREGOSA F, VAROQUAUX G, GRAMFORT A, MICHEL V, THIRION B, GRISEL O, et al. Scikit-learn: Machine Learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.

适用位置：Logistic Regression stacker、AUC/Precision/Recall/F1 等指标计算、交叉验证工具。

[10] PASZKE A, GROSS S, MASSA F, LERER A, BRADBURY J, CHANAN G, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library[C]//Advances in Neural Information Processing Systems. 2019, 32.

适用位置：训练框架、模型推理、CUDA/AMP、张量计算。

## 三、预处理、工程实现与工具依赖

[11] ZUIDERVELD K. Contrast Limited Adaptive Histogram Equalization[M]//HECKBERT P S, ed. Graphics Gems IV. San Diego: Academic Press Professional, 1994: 474-485.

适用位置：CLAHE 对比度增强、超声局部灰度结构增强。

[12] BRADSKI G. The OpenCV Library[J]. Dr. Dobb's Journal of Software Tools, 2000.

适用位置：图像读取、resize、mask 后处理、可视化叠加、CLAHE 实现。

[13] WIGHTMAN R. PyTorch Image Models[EB/OL]. 2019. https://github.com/huggingface/pytorch-image-models

适用位置：timm 模型工厂、预训练权重、模型专属 mean/std、crop_pct 与插值配置。

[14] IAKUBOVSKII P. Segmentation Models PyTorch[EB/OL]. 2019. https://github.com/qubvel-org/segmentation_models.pytorch

适用位置：`segmentation_models_pytorch.Unet`、编码器-解码器分割实现。

[15] VAN DER WALT S, COLBERT S C, VAROQUAUX G. The NumPy Array: A Structure for Efficient Numerical Computation[J]. Computing in Science & Engineering, 2011, 13(2): 22-30. DOI: 10.1109/MCSE.2011.37.

适用位置：数值数组、图像矩阵、指标统计。

[16] MCKINNEY W. Data Structures for Statistical Computing in Python[C]//Proceedings of the 9th Python in Science Conference. 2010: 56-61. DOI: 10.25080/Majora-92bf1922-00a.

适用位置：pandas 表格、预测结果 CSV、实验报告汇总。

[17] ABID A, ABDALLA A, ABID A, KHAN D, ALFOZAN A, ZOU J. Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild[EB/OL]. arXiv:1906.02569, 2019. https://arxiv.org/abs/1906.02569

适用位置：Gradio 本地演示界面、单图上传、结果展示。

## 四、候选模型与对比实验补充引用

[18] HUANG G, LIU Z, VAN DER MAATEN L, WEINBERGER K Q. Densely Connected Convolutional Networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 2261-2269. DOI: 10.1109/CVPR.2017.243.

适用位置：DenseNet121 候选模型、单模型对比、异构集成尝试。

[19] LIU Z, LIN Y, CAO Y, HU H, WEI Y, ZHANG Z, et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 9992-10002. DOI: 10.1109/ICCV48922.2021.00986.

适用位置：Swin-Tiny 候选模型、Transformer 类视觉骨干对比。

[20] SIMONYAN K, ZISSERMAN A. Very Deep Convolutional Networks for Large-Scale Image Recognition[C]//International Conference on Learning Representations. 2015. arXiv:1409.1556.

适用位置：VGG-16 候选基线、早期 CNN 骨干对比。

[21] HOWARD A, SANDLER M, CHU G, CHEN L C, CHEN B, TAN M, et al. Searching for MobileNetV3[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 1314-1324. DOI: 10.1109/ICCV.2019.00140.

适用位置：MobileNetV3-small 候选基线、轻量模型对比。

[22] YOUDEN W J. Index for rating diagnostic tests[J]. Cancer, 1950, 3(1): 32-35. DOI: 10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3.

适用位置：Youden J 阈值选择、Sensitivity 与 Specificity 的综合运行点说明。

[23] WOLPERT D H. Stacked generalization[J]. Neural Networks, 1992, 5(2): 241-259. DOI: 10.1016/S0893-6080(05)80023-1.

适用位置：OOF Logistic Stacking、融合器训练边界、避免直接用外部测试集反向调参。

## 五、报告章节引用映射

| 报告章节 | 建议引用 |
| --- | --- |
| 研究背景与数据来源 | [1], [2] |
| 评价指标与阈值分析 | [3], [9], [22] |
| 分类模型设计 | [4], [5], [7], [10], [13] |
| 模型选择与候选对比 | [4], [5], [18], [19], [20], [21] |
| 病灶分割与 ROI 推理 | [6], [7], [14], [23] |
| 图像预处理 | [11], [12], [13] |
| 可解释性分析 | [8] |
| 工程实现与演示系统 | [10], [12], [15], [16], [17] |

## 六、需要谨慎表述的地方

- BUSI 是外部评估数据集，不能写成训练集或调参集。
- BUSBRA 官方许可和赛题文本都指向 BUS-BRA 文献，正式报告应保留 [1]。
- 若报告篇幅有限，工程依赖可合并为“本系统基于 PyTorch、OpenCV、scikit-learn、Gradio 等开源工具实现”，并只在参考文献中保留 [9], [10], [12], [17]。
- 若学校要求严格 GB/T 7714 格式，建议导入 Zotero/NoteExpress 后再统一大小写、标点和英文作者缩写。
