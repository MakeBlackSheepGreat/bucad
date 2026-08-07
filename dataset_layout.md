# 数据集目录与职责

```text
训练集/
  BUSBRA/                         训练与内部 OOF；保留原始官方目录

测试集/
  Dataset_BUSI_with_GT/           既有 BUSI 外部评估；保留原始官方目录

data/external/
  bus_uclm/                       独立外部测试；images、masks、manifest.csv
  busi_whu/
    raw/                          原始图像和分割标注
    label_reference_hf/           标签映射参考副本
    manifests/                    经过 SHA-256 一致性校验的 788 例外部 manifest
  tcia_breast_us/
    downloads/                    TCIA 原始 ZIP 与临床 XLSX
    images_and_masks/             解压后的 PNG 图像和肿瘤 mask
    manifests/                    临床表 `Classification` 对齐后的 252 例 manifest

data/catalog/                     自动生成的统一 manifest 与数据卡
```

## 固定职责

| 队列 | 角色 | 样本协议 |
| --- | --- | --- |
| BUSBRA | 训练、内部 OOF | 所有模型选择与阈值选择只在此阶段完成 |
| BUSI | 外部评估 | binary：benign vs malignant，normal 不进入主二分类表 |
| BUS-UCLM | 外部评估 | binary：174 benign、90 malignant，normal 排除 |
| BUSI-WHU | 外部评估 | 788 例可复核标签，原图与标签参考通过 SHA-256 对齐 |
| TCIA BrEaST | 外部评估 | `Classification` 的 benign/malignant，4 个 normal 不进入主二分类表 |

`scripts/build_dataset_catalog.py` 会重建 `data/catalog/`。原始数据和自动生成目录受 `.gitignore` 保护。
