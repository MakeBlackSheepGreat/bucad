# SonoGloRe-ConvNeXt V1 分支整理

日期：2026-06-20

## 分支目标

当前分支的主目标是固定并评估一个现代 CNN 单模型 `SonoGloRe-ConvNeXt V1`，比较对象是 `ConvNeXt-Tiny`。本分支的核心评估问题是架构是否比基础 ConvNeXt 更有表达力，而非 demo 主线 ensemble 的最高 AUC。

## 当前分支改动分组

### 1. V1 模型冻结

- 模型别名：`sonoglore_convnext_v1`
- 代码位置：`src/models/classifier.py`
- BUSBRA 训练配置：`configs/classifier/sonoglore_convnext_v1.yml`
- BUSI 推理配置：
  - `configs/inference/sonoglore_convnext_v1_5fold_tta_crop_sweep.yml`
  - `configs/inference/sonoglore_convnext_v1_existing_5fold_tta_crop_sweep.yml`
- 冻结说明：`artifacts/reports/generated/sonoglore_convnext_v1_frozen_protocol.md`

固定结构：

| 模块 | V1 设置 |
| --- | --- |
| backbone | `convnext_tiny` |
| active stages | `[2, 3, 4]` |
| stage projection dims | `[192, 256, 320]` |
| stage fusion weights | `[0.8, 1.0, 1.0]` |
| projection block | `1x1 projection + GELU + GRN + ECA` |
| stage4 attention | off |
| scale gate | off |

已有 BUSI 结果记录：

| 模型 | BUSI AUC | Sensitivity | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny 5fold crop-sweep | 0.9054 | 0.8048 | 0.8719 | 0.7770 |
| SonoGloRe-ConvNeXt V1 5fold crop-sweep | 0.9071 | 0.8190 | 0.8719 | 0.7854 |

### 2. ImageNet 公平实验

- 核心实现：`src/experiments/imagenet.py`
- CLI：
  - `scripts/prepare_imagenet.py`
  - `scripts/train_imagenet.py`
  - `scripts/eval_imagenet.py`
  - `scripts/run_imagenet_ablation.py`
- 配置目录：`configs/imagenet/`
- 设置说明：`artifacts/reports/generated/imagenet_fair_eval_setup_2026-06-19.md`

当前本机 ImageNet 根目录配置为 `D:\projects\ImageNet`。需要下载到该目录的官方文件：

| 文件 | 用途 | MD5 |
| --- | --- | --- |
| `ILSVRC2012_devkit_t12.tar.gz` | Task 1 & 2 devkit | `fa75699e90414af021442c21a62c3abf` |
| `ILSVRC2012_img_train.tar` | Task 1 & 2 training images | `1d675b47d978889d74fa0da5fadfb00e` |
| `ILSVRC2012_img_val.tar` | validation images | `29b22e2961454d5413ddabcf34fc5622` |

不要把 V1 的 `pretrained: true` backbone-only 结果当作公平 ImageNet 结果。V1 新增 projection、GRN/ECA、融合头和分类头都需要完整 ImageNet checkpoint。

### 3. ROI dual-view 探索

- 训练配置：`configs/classifier/roi_dualview_convnext_tiny.yml`
- 推理配置：
  - `configs/inference/roi_dualview_convnext_tiny_fold1.yml`
  - `configs/inference/roi_dualview_convnext_tiny_fold1_tta_crop_sweep.yml`
- 主要代码：
  - `src/datasets/busbra.py`
  - `src/models/classifier.py`
  - `src/engine/train_cls.py`
  - `src/engine/classifier_ensemble.py`
  - `src/engine/inference.py`

这部分来自 demo 主线启发，但当前分支主线已转向 V1/ImageNet 架构公平性测试。ROI dual-view 可保留为后续医学迁移实验，不作为当前 ImageNet 对照的一部分。

### 4. 报告与阈值分析

- 新增 BUSI 结果：
  - `artifacts/reports/generated/busi_sonoglore_convnext_tiny_grn_eca_asymproj_posw16_5fold.json`
  - `artifacts/reports/generated/busi_sonoglore_convnext_tiny_grn_eca_asymproj_posw17_5fold_tta_crop_sweep.json`
- 新增阈值分析：
  - `artifacts/reports/generated/threshold_analysis_busi_sonoglore_convnext_tiny_grn_eca_asymproj_posw16_5fold.md`
  - `artifacts/reports/generated/threshold_analysis_busi_sonoglore_convnext_tiny_grn_eca_asymproj_posw17_5fold_tta_crop_sweep.md`

## 下载完成后的执行顺序

先确认三份官方归档已经放到 `D:\projects\ImageNet`。

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/prepare_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_train.yml --prepare-train --prepare-val
```

可先做 1 batch smoke：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_train.yml --epochs-override 1 --max-batches 1 --max-val-batches 1
```

正式公平训练：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/convnext_tiny_train.yml
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_train.yml
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_no_eca_train.yml
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_no_grn_train.yml
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/train_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_uniform_projection_train.yml
```

统一消融评估：

```powershell
& "$env:USERPROFILE\.conda\envs\BUCAD\python.exe" scripts/run_imagenet_ablation.py --config configs/imagenet/sonoglore_convnext_v1_ablation.yml
```

## 当前验证状态

已通过：

```text
python -m pytest tests/unit/test_imagenet_experiment.py tests/unit/test_classifier_models.py::test_sonoglore_convnext_v1_alias_freezes_selected_structure tests/unit/test_config.py::test_load_project_config_resolves_paths_relative_to_paths_file -q
14 passed
```

已通过：

```text
python -m compileall src scripts tests
```

当前阻塞项：

- 本机 ImageNet 根目录已切到 `D:\projects\ImageNet`。
- `ILSVRC2012_devkit_t12.tar.gz` 已存在。
- `ILSVRC2012_img_val.tar` 已解包整理完成，`torchvision.datasets.ImageNet(..., split="val")` 可加载 `50000` 张验证图、`1000` 类。
- `ILSVRC2012_img_train.tar` 当前文件大小为 `79027683328` bytes，tar 枚举到第 `523` 个内部类别归档后报 `ReadError unexpected end of data`，需要重新下载或续传完整 train tar 后才能准备训练集。
- 正式 300 epoch ImageNet 训练需要较长 GPU 时间；建议先用 smoke 命令确认数据准备和训练闭环。

## 建议提交分组

1. `model: freeze sonoglore convnext v1`
   - `src/models/classifier.py`
   - `configs/classifier/sonoglore_convnext_v1.yml`
   - `configs/inference/sonoglore_convnext_v1*.yml`
   - `tests/unit/test_classifier_models.py`
   - `artifacts/reports/generated/sonoglore_convnext_v1_frozen_protocol.md`

2. `experiments: add imagenet fair evaluation pipeline`
   - `src/experiments/imagenet.py`
   - `scripts/prepare_imagenet.py`
   - `scripts/train_imagenet.py`
   - `scripts/eval_imagenet.py`
   - `scripts/run_imagenet_ablation.py`
   - `configs/imagenet/`
   - `src/utils/paths.py`
   - `configs/paths.example.yml`
   - `configs/paths.local.yml`
   - `tests/unit/test_imagenet_experiment.py`
   - `tests/unit/test_config.py`
   - `artifacts/reports/generated/imagenet_fair_eval_setup_2026-06-19.md`

3. `experiment: add roi dual-view classifier path`
   - ROI dual-view dataset/model/train/inference files and tests.

4. `reports: add sonoglore busi threshold analyses`
   - Generated BUSI JSON and threshold Markdown files.
