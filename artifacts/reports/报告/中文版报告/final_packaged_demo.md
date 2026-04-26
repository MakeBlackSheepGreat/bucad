<!-- 本文件由英文版报告同步翻译生成；模型名、数据集名、路径、命令、指标名等专有名词按原文保留。 -->

# 最终打包 Demo 冒烟测试

日期：2026-04-24

## 结果

- 命令：从 `dist/bucad-demo` 启动 `dist/bucad-demo/bucad-demo.exe`。
- 冒烟测试时长：20 秒。
- 结果：PASS，进程保持运行 20 秒，随后由验证脚本停止。
- Stdout：`artifacts/reports/packaged_demo_stdout.txt`。

## 备注

- 本测试只证明打包程序可以启动并保持运行，不证明医学结果正确。
- 正式展示前仍需人工打开浏览器执行单图上传流程。
