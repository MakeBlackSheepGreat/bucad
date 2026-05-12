from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busbra import generate_busbra_split_assignments, load_busbra_manifest
from src.datasets.busi import load_busi_manifest
from src.utils.config import load_yaml
from src.utils.metrics import best_threshold_by_youden, classification_metrics, threshold_sweep
from src.utils.paths import ProjectPaths
from src.utils.runtime import ensure_dir, seed_everything


LABEL_TO_INDEX = {"benign": 0, "malignant": 1}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _load_or_create_splits(manifest: pd.DataFrame, split_path: Path, *, fold_count: int, seed: int) -> pd.DataFrame:
    if split_path.exists():
        return pd.read_csv(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    assignments = generate_busbra_split_assignments(manifest, n_splits=fold_count, seed=seed)
    assignments.to_csv(split_path, index=False)
    return assignments


def _split_manifest_for_fold(
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_assignments = assignments[assignments["fold_id"] == fold]
    train_ids = set(fold_assignments.loc[fold_assignments["stage"] == "train", "sample_id"])
    val_ids = set(fold_assignments.loc[fold_assignments["stage"] == "val", "sample_id"])
    train_manifest = manifest[manifest["sample_id"].isin(train_ids)].reset_index(drop=True)
    val_manifest = manifest[manifest["sample_id"].isin(val_ids)].reset_index(drop=True)
    return train_manifest, val_manifest


def _materialize_yolo_split(frame: pd.DataFrame, destination: Path) -> None:
    for row in frame.itertuples(index=False):
        label = str(row.pathology_label).lower()
        source = Path(row.image_path)
        target = destination / label / source.name
        _safe_link_or_copy(source, target)


def _build_yolo_dataset(
    *,
    data_root: Path,
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
) -> Path:
    for split_name, frame in (("train", train_manifest), ("val", val_manifest)):
        _materialize_yolo_split(frame, data_root / split_name)
    return data_root


def _evaluate_manifest(
    model,
    manifest: pd.DataFrame,
    *,
    imgsz: int,
    batch: int,
    device: str,
    threshold: float,
) -> dict[str, Any]:
    names = {int(key): str(value).lower() for key, value in model.names.items()}
    malignant_indices = [idx for idx, name in names.items() if name == "malignant"]
    if not malignant_indices:
        raise ValueError(f"Cannot find malignant class in YOLO names: {model.names}")
    malignant_index = malignant_indices[0]

    paths = [str(Path(path)) for path in manifest["image_path"].tolist()]
    y_true: list[int] = []
    malignant_probabilities: list[float] = []
    rows: list[dict[str, Any]] = []
    results = model.predict(
        source=paths,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        stream=True,
    )
    for row, result in zip(manifest.itertuples(index=False), results):
        probs = result.probs.data.detach().cpu().numpy()
        malignant_probability = float(probs[malignant_index])
        label = str(row.pathology_label).lower()
        target = int(LABEL_TO_INDEX[label])
        y_true.append(target)
        malignant_probabilities.append(malignant_probability)
        rows.append(
            {
                "sample_id": str(row.sample_id),
                "pathology_label": label,
                "image_path": str(row.image_path),
                "malignant_probability": malignant_probability,
            }
        )

    metrics = classification_metrics(y_true, malignant_probabilities, threshold=threshold)
    best = best_threshold_by_youden(y_true, malignant_probabilities)
    sweep = threshold_sweep(y_true, malignant_probabilities)
    return {
        "sample_count": len(rows),
        "metrics_at_threshold": metrics,
        "best_by_youden": best,
        "threshold_sweep": sweep,
        "rows": rows,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _confusion_text(metrics: dict[str, Any]) -> str:
    confusion = metrics.get("confusion", {})
    return (
        f"TN {confusion.get('tn', 0)} / FP {confusion.get('fp', 0)} / "
        f"FN {confusion.get('fn', 0)} / TP {confusion.get('tp', 0)}"
    )


def _markdown_report(report: dict[str, Any]) -> list[str]:
    busi = report["evaluation"]["busi_external"]
    busbra = report["evaluation"]["busbra_fold1_val"]
    busi_default = busi["metrics_at_threshold"]
    busi_best = busi["best_by_youden"]
    busbra_default = busbra["metrics_at_threshold"]
    busbra_best = busbra["best_by_youden"]
    lines = [
        "# YOLO Fold1 单模型对比实验",
        "",
        "## 实验边界",
        "",
        "- 本实验为旁路模型筛选，不修改 `configs/inference/demo.yml`。",
        "- 训练仅使用 BUSBRA fold1 train；BUSBRA fold1 val 用于内部复核。",
        "- BUSI 只做当前冻结单模型的一次外部验证，不用于阈值回调或二次调参。",
        "- 本次只测试一个 YOLO 分类候选模型。",
        "",
        "## 模型选择依据",
        "",
        f"- YOLO 候选：`{report['model']['requested_model']}`。",
        "- 选择依据：Ultralytics 官方 YOLO 分类模型表中，x 规模分类模型是同一 YOLO26 分类族内 Top-1 精度最高的公开权重。",
        "- 任务适配：按图像分类方式训练 benign/malignant 二分类，不使用检测框或分割标签。",
        "",
        "## 训练设置",
        "",
        f"- fold：`{report['fold']}`",
        f"- epochs：`{report['training']['epochs']}`",
        f"- imgsz：`{report['training']['imgsz']}`",
        f"- batch：`{report['training']['batch']}`",
        f"- device：`{report['training']['device']}`",
        f"- 训练样本：`{report['data']['train_count']}`",
        f"- BUSBRA val 样本：`{report['data']['val_count']}`",
        f"- BUSI 样本：`{busi['sample_count']}`",
        f"- best checkpoint：`{report['training']['best_checkpoint']}`",
        "",
        "## 结果",
        "",
        "| 数据集 | 阈值 | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Confusion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| BUSBRA fold1 val | {busbra_default['threshold']:.2f} | {_fmt(busbra_default['auc'])} | "
            f"{_fmt(busbra_default['accuracy'])} | {_fmt(busbra_default['sensitivity'])} | "
            f"{_fmt(busbra_default['specificity'])} | {_fmt(busbra_default['precision'])} | "
            f"{_fmt(busbra_default['f1_score'])} | {_confusion_text(busbra_default)} |"
        ),
        (
            f"| BUSI external | {busi_default['threshold']:.2f} | {_fmt(busi_default['auc'])} | "
            f"{_fmt(busi_default['accuracy'])} | {_fmt(busi_default['sensitivity'])} | "
            f"{_fmt(busi_default['specificity'])} | {_fmt(busi_default['precision'])} | "
            f"{_fmt(busi_default['f1_score'])} | {_confusion_text(busi_default)} |"
        ),
        "",
        "## Youden 最优点（仅报告，不回调参数）",
        "",
        "| 数据集 | 阈值 | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Youden J | Confusion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| BUSBRA fold1 val | {busbra_best['threshold']:.2f} | {_fmt(busbra_best['accuracy'])} | "
            f"{_fmt(busbra_best['sensitivity'])} | {_fmt(busbra_best['specificity'])} | "
            f"{_fmt(busbra_best['precision'])} | {_fmt(busbra_best['f1_score'])} | "
            f"{_fmt(busbra_best['youden_j'])} | {_confusion_text(busbra_best)} |"
        ),
        (
            f"| BUSI external | {busi_best['threshold']:.2f} | {_fmt(busi_best['accuracy'])} | "
            f"{_fmt(busi_best['sensitivity'])} | {_fmt(busi_best['specificity'])} | "
            f"{_fmt(busi_best['precision'])} | {_fmt(busi_best['f1_score'])} | "
            f"{_fmt(busi_best['youden_j'])} | {_confusion_text(busi_best)} |"
        ),
        "",
        "## Fold1 参考基线（BUSI 外部，默认阈值 0.50）",
        "",
        "| 模型 | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| ConvNeXt-Small timm recipe fold1 | 0.8947 | 0.8284 | 0.7667 | 0.8581 | 0.7220 | 0.7436 |",
        "| ConvNeXt-Tiny timm recipe fold1 | 0.8943 | 0.8423 | 0.7762 | 0.8741 | 0.7477 | 0.7617 |",
        "| DenseNet121 fold1 | 0.8766 | 0.8083 | 0.4571 | 0.9771 | 0.9057 | 0.6076 |",
        "| Swin-Tiny timm recipe fold1 | 0.8729 | 0.8300 | 0.7048 | 0.8902 | 0.7551 | 0.7291 |",
        "| EfficientNetV2-S fold1 | 0.8609 | 0.7465 | 0.8333 | 0.7048 | 0.5757 | 0.6809 |",
        (
            f"| {report['model']['trained_model_name']} fold1 | {_fmt(busi_default['auc'])} | "
            f"{_fmt(busi_default['accuracy'])} | {_fmt(busi_default['sensitivity'])} | "
            f"{_fmt(busi_default['specificity'])} | {_fmt(busi_default['precision'])} | "
            f"{_fmt(busi_default['f1_score'])} |"
        ),
        "",
        "## 初步结论",
        "",
    ]
    busi_auc = float(busi_default.get("auc") or 0.0)
    if busi_auc > 0.8947:
        lines.append("- YOLO fold1 在 BUSI AUC 上超过既有 fold1 单模型基线，建议后续进入更严格的五折与 OOF 协议。")
    else:
        lines.append("- YOLO fold1 未超过既有 ConvNeXt 单折基线，暂不建议进入主线，仅保留为旁路实验记录。")
    lines.extend(
        [
            "- 该结果只代表单折模型基础潜力，不代表五折集成或 ROI-aware 主线性能。",
            "",
        ]
    )
    return lines


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO
    import ultralytics

    seed_everything(args.seed)
    root = _repo_root()
    paths_config = root / args.paths_config
    paths = ProjectPaths.from_mapping(load_yaml(paths_config), config_path=paths_config)
    manifest = load_busbra_manifest(paths.busbra_root)
    split_path = root / args.split_path
    assignments = _load_or_create_splits(manifest, split_path, fold_count=args.fold_count, seed=args.seed)
    train_manifest, val_manifest = _split_manifest_for_fold(manifest, assignments, args.fold)
    if train_manifest.empty or val_manifest.empty:
        raise RuntimeError(f"Fold {args.fold} produced empty train/val split.")

    data_root = root / args.data_root
    _build_yolo_dataset(
        data_root=data_root,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
    )

    project_dir = root / args.project_dir
    run_name = args.run_name or f"{Path(args.model).stem}_fold{args.fold}"
    model = YOLO(args.model)
    train_result = model.train(
        data=str(data_root),
        task="classify",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        seed=args.seed,
        workers=args.workers,
        plots=False,
        verbose=True,
    )
    save_dir = Path(getattr(train_result, "save_dir", project_dir / run_name))
    best_checkpoint = save_dir / "weights" / "best.pt"
    if not best_checkpoint.exists():
        best_checkpoint = save_dir / "weights" / "last.pt"
    trained_model = YOLO(str(best_checkpoint))

    busi_manifest = load_busi_manifest(paths.busi_root, include_normal=False)
    busbra_eval = _evaluate_manifest(
        trained_model,
        val_manifest,
        imgsz=args.imgsz,
        batch=args.eval_batch,
        device=args.device,
        threshold=args.threshold,
    )
    busi_eval = _evaluate_manifest(
        trained_model,
        busi_manifest,
        imgsz=args.imgsz,
        batch=args.eval_batch,
        device=args.device,
        threshold=args.threshold,
    )

    report = {
        "method_id": f"yolo_cls_{Path(args.model).stem}_fold{args.fold}",
        "created_by": "scripts/run_yolo_cls_fold1_experiment.py",
        "fold": args.fold,
        "data_boundary": {
            "training": "BUSBRA fold1 train",
            "internal_validation": "BUSBRA fold1 val",
            "external_review": "BUSI one-pass frozen review",
            "busi_tuning": False,
        },
        "model": {
            "requested_model": args.model,
            "trained_model_name": Path(args.model).stem,
            "ultralytics_version": ultralytics.__version__,
        },
        "data": {
            "train_count": int(len(train_manifest)),
            "val_count": int(len(val_manifest)),
            "busi_count": int(len(busi_manifest)),
            "data_root": str(data_root),
            "split_path": str(split_path),
        },
        "training": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "eval_batch": args.eval_batch,
            "device": args.device,
            "workers": args.workers,
            "save_dir": str(save_dir),
            "best_checkpoint": str(best_checkpoint),
        },
        "evaluation": {
            "busbra_fold1_val": busbra_eval,
            "busi_external": busi_eval,
        },
    }
    report_dir = root / args.report_dir
    ensure_dir(report_dir)
    json_path = report_dir / f"{report['method_id']}.json"
    md_path = report_dir / f"{report['method_id']}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(_markdown_report(report)) + "\n", encoding="utf-8")
    report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-fold YOLO classification comparison.")
    parser.add_argument("--model", default="yolo26x-cls.pt")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--paths-config", default="configs/paths.local.yml")
    parser.add_argument("--split-path", default="artifacts/reports/busbra_5fold_splits.csv")
    parser.add_argument("--data-root", default="artifacts/datasets/yolo_cls_fold1")
    parser.add_argument("--project-dir", default="artifacts/yolo_runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--report-dir",
        default="artifacts/reports/Chinese reports/01_baseline_model_screening",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_experiment(args)
    print(report["report_paths"]["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
