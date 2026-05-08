# BUCAD v1.0.0 Windows Demo 打包记录

日期：2026-04-26

## 版本定位

`v1.0.0` 是比赛展示用 Windows demo 版本，目标是让评委或队友通过一个可执行文件启动本地 Gradio 诊断界面。

## 打包命令

```powershell
conda activate BUCAD
python -m PyInstaller --clean --noconfirm packaging\demo.spec
```

## 输出位置

| 项目 | 路径 |
| --- | --- |
| 可执行文件 | `dist/bucad-demo/bucad-demo.exe` |
| 打包配置 | `packaging/demo.spec` |
| 内置配置 | `dist/bucad-demo/_internal/configs/inference/demo.yml` |
| 内置权重目录 | `dist/bucad-demo/_internal/artifacts/checkpoints/` |

## 内置模型资产

- ConvNeXt-Tiny 五折 checkpoint：5 个。
- EfficientNetV2-S 五折 checkpoint：5 个。
- Segmenter checkpoint：1 个。
- 合计 checkpoint：11 个。
- checkpoint 体积约 `0.952 GB`。
- 完整发布目录体积约 `4.22 GB`，主要来自 PyTorch/CUDA 运行库和模型权重。

## 烟测结果

- `bucad-demo.exe` 可以从 `dist/bucad-demo/` 后台启动。
- 本地 Gradio 页面 `http://127.0.0.1:7860/` 返回 HTTP 200。
- 启动入口已设置为自动打开浏览器。

## 重要说明

- `dist/` 和 `artifacts/releases/` 不提交到 Git 仓库，避免把数 GB 二进制文件直接塞进源码仓库。
- 如果需要在 GitHub Release 附加二进制包，应使用 release asset；如果单个压缩包超过 GitHub 单文件限制，应改用外部网盘或分卷发布。
- BUSI 仍仅用于外部评估；demo 固定阈值来自 BUSBRA OOF/ROI 流程，不应在发布时用 BUSI 重新调参。
