# ImageNet 公平评估设置记录

日期：2026-06-19

## 目的

本记录服务于 SonoGloRe-ConvNeXt V1 的架构公平性评估。ImageNet 实验用于模仿 ConvNeXt 论文的通用视觉分类评估与模块消融，不用于 demo 主线性能优化。

## 已新增入口

- 数据准备：`scripts/prepare_imagenet.py`
- 单模型训练：`scripts/train_imagenet.py`
- 单模型评估：`scripts/eval_imagenet.py`
- 多模型消融：`scripts/run_imagenet_ablation.py`

## 已新增配置

- ConvNeXt-Tiny 训练：`configs/imagenet/convnext_tiny_train.yml`
- ConvNeXt-Tiny 评估：`configs/imagenet/convnext_tiny_eval.yml`
- SonoGloRe-ConvNeXt V1 训练：`configs/imagenet/sonoglore_convnext_v1_train.yml`
- SonoGloRe-ConvNeXt V1 评估：`configs/imagenet/sonoglore_convnext_v1_eval.yml`
- V1 消融训练：
  - `configs/imagenet/sonoglore_convnext_v1_no_eca_train.yml`
  - `configs/imagenet/sonoglore_convnext_v1_no_grn_train.yml`
  - `configs/imagenet/sonoglore_convnext_v1_uniform_projection_train.yml`
- V1 模块消融：`configs/imagenet/sonoglore_convnext_v1_ablation.yml`

## 公平性边界

不能把 `sonoglore_convnext_v1` 的 `pretrained: true` 直接评估结果视为 ImageNet 公平结果。V1 的 ConvNeXt backbone 可以加载 ImageNet 预训练，但新增的多尺度投影、GRN/ECA、融合头和分类头没有对应预训练权重。公平对照必须先训练完整 V1 checkpoint，再用同一验证协议统计 Top-1、Top-5、吞吐量和参数量。

ConvNeXt-Tiny 可用 timm 官方预训练权重做 sanity check；严格论文式对照则使用 `configs/imagenet/convnext_tiny_train.yml` 从相同 recipe 训练本地 checkpoint。

## 训练 recipe

配置模仿 ConvNeXt CVPR 2022 Appendix A.1 的 ImageNet-1K 训练设置：

| 项目 | 设置 |
| --- | --- |
| 输入 | 224 |
| optimizer | AdamW |
| base lr | 0.004，按 batch size / 4096 线性缩放 |
| weight decay | 0.05 |
| epochs | 300 |
| warmup | 20 epochs linear warmup |
| scheduler | cosine decay |
| augmentation | RandAugment、Mixup 0.8、CutMix 1.0、Random Erasing 0.25 |
| regularization | label smoothing 0.1 |
| eval | center crop、Top-1、Top-5、images/s、params |

## 当前本机数据状态

配置的 ImageNet 根目录：

`C:\Users\876762330\Desktop\projects\Agent\datasets\imagenet`

当前缺少官方 ImageNet-1K ILSVRC2012 归档：

- `ILSVRC2012_devkit_t12.tar.gz`
- `ILSVRC2012_img_val.tar`
- `ILSVRC2012_img_train.tar`

脚本会校验 torchvision 使用的官方 MD5：

| 文件 | MD5 |
| --- | --- |
| `ILSVRC2012_img_train.tar` | `1d675b47d978889d74fa0da5fadfb00e` |
| `ILSVRC2012_img_val.tar` | `29b22e2961454d5413ddabcf34fc5622` |
| `ILSVRC2012_devkit_t12.tar.gz` | `fa75699e90414af021442c21a62c3abf` |

## 官方访问边界

ImageNet-1K 不是可匿名自动下载的数据集。需要先通过官方 ImageNet/ILSVRC 条款或其他已授权渠道取得上述归档，再放入配置的根目录。仓库只提供准备、校验、评估和消融脚本。

## 后续命令

准备验证集：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/prepare_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_eval.yml --prepare-val
```

准备训练集与验证集：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/prepare_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_train.yml --prepare-train --prepare-val
```

训练 ConvNeXt-Tiny 对照：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/convnext_tiny_train.yml
```

训练 SonoGloRe-ConvNeXt V1：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_train.yml
```

训练 V1 消融：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_no_eca_train.yml
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_no_grn_train.yml
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_uniform_projection_train.yml
```

评估 ConvNeXt-Tiny sanity baseline：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/eval_imagenet.py --config configs/imagenet/convnext_tiny_eval.yml
```

评估 SonoGloRe-ConvNeXt V1 完整 checkpoint：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/eval_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_eval.yml
```

运行 V1 消融：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/run_imagenet_ablation.py --config configs/imagenet/sonoglore_convnext_v1_ablation.yml
```

快速 smoke：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_train.yml --epochs-override 1 --max-batches 1 --max-val-batches 1
```
