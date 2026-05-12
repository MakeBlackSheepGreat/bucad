from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.train_seg import run_segmentation_training  # noqa: E402
from src.utils.config import load_project_config  # noqa: E402
from src.utils.reporting import write_json_report, write_markdown_report  # noqa: E402
from src.utils.runtime import timestamp_now  # noqa: E402


REPORT_DIR = Path("artifacts/reports/Chinese reports/07_segmenter_architecture_replacement")
MAINLINE_METRICS = {
    "auc": 0.9255802549852892,
    "threshold": 0.51,
    "accuracy": 0.8531684698608965,
    "sensitivity": 0.8666666666666667,
    "specificity": 0.8466819221967964,
    "precision": 0.7309236947791165,
    "f1_score": 0.7930283224400873,
}


@dataclass(frozen=True)
class SegmenterMethod:
    method_id: str
    category: str
    rationale: str
    model: dict[str, Any]
    loss: dict[str, Any]


METHODS: tuple[SegmenterMethod, ...] = (
    SegmenterMethod(
        method_id="unetplusplus_densenet121_bce_dice",
        category="U-Net++ / DenseNet121",
        rationale="Use nested U-Net++ skip fusion with the DenseNet121 encoder reported as a strong BUSI backbone direction.",
        model={
            "architecture": "unetplusplus",
            "encoder_name": "densenet121",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    SegmenterMethod(
        method_id="unetplusplus_resnet34_bce_dice",
        category="U-Net++ / stronger ResNet",
        rationale="Keep the U-Net++ decoder while moving from ResNet18 to a deeper ResNet34 encoder.",
        model={
            "architecture": "unetplusplus",
            "encoder_name": "resnet34",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    SegmenterMethod(
        method_id="unetplusplus_efficientnetb0_bce_dice",
        category="U-Net++ / EfficientNet",
        rationale="Test whether a compact compound-scaled encoder gives cleaner ROI masks than ResNet18 under the same ROI protocol.",
        model={
            "architecture": "unetplusplus",
            "encoder_name": "efficientnet-b0",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    SegmenterMethod(
        method_id="fpn_efficientnetb0_bce_dice",
        category="multi-scale feature pyramid",
        rationale="Use FPN to aggregate multi-resolution encoder features, matching the context and scale-robustness direction in BUS segmentation papers.",
        model={
            "architecture": "fpn",
            "encoder_name": "efficientnet-b0",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    SegmenterMethod(
        method_id="fpn_densenet121_bce_dice",
        category="multi-scale DenseNet",
        rationale="Combine the DenseNet121 encoder with a feature pyramid decoder as a medium-complexity alternative to U-Net++.",
        model={
            "architecture": "fpn",
            "encoder_name": "densenet121",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    SegmenterMethod(
        method_id="deeplabv3plus_resnet34_bce_dice",
        category="atrous context / DeepLabV3+",
        rationale="Use ASPP-style dilated context to test whether broader ultrasound texture context improves downstream ROI classification.",
        model={
            "architecture": "deeplabv3plus",
            "encoder_name": "resnet34",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    SegmenterMethod(
        method_id="manet_resnet34_bce_dice",
        category="attention decoder",
        rationale="Use MAnet attention blocks as a practical Attention U-Net style substitute available in the local SMP stack.",
        model={
            "architecture": "manet",
            "encoder_name": "resnet34",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    SegmenterMethod(
        method_id="linknet_densenet121_bce_dice",
        category="light decoder / DenseNet",
        rationale="Test a lighter decoder with DenseNet121 to see if simpler masks transfer better to the fixed ROI gate.",
        model={
            "architecture": "linknet",
            "encoder_name": "densenet121",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    SegmenterMethod(
        method_id="pan_resnet34_bce_dice",
        category="pyramid attention",
        rationale="Use PAN as a pyramid-attention segmentation baseline, covering the non-SAM attention/context family.",
        model={
            "architecture": "pan",
            "encoder_name": "resnet34",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    SegmenterMethod(
        method_id="pspnet_resnet34_bce_dice",
        category="pyramid pooling context",
        rationale="Use PSPNet pyramid pooling as another context-expansion baseline without introducing SAM-style prompt dependencies.",
        model={
            "architecture": "pspnet",
            "encoder_name": "resnet34",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train non-SAM segmenter replacements and run one frozen BUSI review for each."
    )
    parser.add_argument("--base-config", default="configs/segmenter/unet_5fold.yml")
    parser.add_argument("--runtime-config", default="configs/inference/demo.yml")
    parser.add_argument("--methods", default="all", help="Comma-separated method ids, or all.")
    parser.add_argument("--fold-count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-busi", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retrain and re-evaluate even when reports already exist.")
    parser.add_argument("--output", default=str(REPORT_DIR / "segmenter_architecture_replacement_protocol.json"))
    parser.add_argument("--markdown", default=str(REPORT_DIR / "segmenter_architecture_replacement_protocol.md"))
    return parser


def _selected_methods(value: str) -> list[SegmenterMethod]:
    if value == "all":
        return list(METHODS)
    selected = {item.strip() for item in value.split(",") if item.strip()}
    method_by_id = {method.method_id: method for method in METHODS}
    missing = sorted(selected - set(method_by_id))
    if missing:
        raise ValueError(f"Unknown method ids: {missing}")
    return [method_by_id[method_id] for method_id in selected]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _write_yaml(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return target


def _method_config(
    *,
    base_config: dict[str, Any],
    method: SegmenterMethod,
    output_root: Path,
    fold_count: int,
    device: str | None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["model"] = copy.deepcopy(method.model)
    config["loss"] = copy.deepcopy(method.loss)
    if device:
        config["device"] = device
    config.setdefault("training", {})
    config["training"]["fold_count"] = int(fold_count)
    config.setdefault("output", {})
    config["output"]["checkpoint_name"] = f"segmenter_arch_{method.method_id}_fold{{fold}}.pt"
    config["output"]["report_name"] = f"segmenter_arch_{method.method_id}_train_fold{{fold}}.json"
    config["segmenter_architecture_replacement"] = {
        "method_id": method.method_id,
        "category": method.category,
        "rationale": method.rationale,
        "generated_at": timestamp_now(),
        "boundary": "BUSBRA training only; BUSI is a frozen one-pass external review.",
    }
    config["report_dir"] = str(output_root)
    return config


def _expected_training_report_path(paths, config: dict[str, Any], fold: int) -> Path:
    report_name = config.get("output", {}).get("report_name", "train_seg_fold{fold}.json").format(fold=fold)
    return paths.reports_root / report_name


def _expected_checkpoint_path(paths, config: dict[str, Any], fold: int) -> Path:
    checkpoint_name = config.get("output", {}).get("checkpoint_name", "segmenter_fold{fold}.pt").format(fold=fold)
    return paths.checkpoints_root / checkpoint_name


def _load_cached_fold_report(paths, config: dict[str, Any], fold: int) -> dict[str, Any] | None:
    report_path = _expected_training_report_path(paths, config, fold)
    checkpoint_path = _expected_checkpoint_path(paths, config, fold)
    if not report_path.exists() or not checkpoint_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["checkpoint_path"] = str(checkpoint_path)
    report["cached"] = True
    return report


def _mean_metrics(fold_reports: list[dict[str, Any]]) -> dict[str, float]:
    metric_keys = sorted(
        {
            key
            for report in fold_reports
            for key, value in report.get("metrics", {}).items()
            if isinstance(value, (int, float)) and np.isfinite(float(value))
        }
    )
    return {
        key: float(np.mean([float(report["metrics"][key]) for report in fold_reports if key in report["metrics"]]))
        for key in metric_keys
    }


def _freeze_runtime_config(
    *,
    runtime_config_path: str | Path,
    method: SegmenterMethod,
    fold_reports: list[dict[str, Any]],
    output_root: Path,
) -> Path:
    runtime = _load_yaml(runtime_config_path)
    checkpoints = [report["checkpoint_path"] for report in sorted(fold_reports, key=lambda item: item["fold"])]
    runtime.setdefault("runtime", {})
    if len(checkpoints) == 1:
        runtime["runtime"]["segmenter_checkpoint"] = checkpoints[0]
        runtime["runtime"].pop("segmenter_checkpoints", None)
    else:
        runtime["runtime"]["segmenter_checkpoint"] = None
        runtime["runtime"]["segmenter_checkpoints"] = checkpoints
    runtime["runtime"]["ensemble_display_name"] = (
        f"{runtime['runtime'].get('ensemble_display_name', 'BUCAD')} + {method.method_id}"
    )
    runtime["segmenter_architecture_replacement"] = {
        "method_id": method.method_id,
        "category": method.category,
        "frozen_at": timestamp_now(),
        "note": "Only the segmenter checkpoint is replaced; classifier, threshold, ROI stacker, and ROI area gate stay fixed.",
    }
    return _write_yaml(output_root / "frozen_configs" / f"demo_{method.method_id}.yml", runtime)


def _run_busi_review(config_path: Path, output_root: Path, method_id: str, *, force: bool) -> dict[str, Any]:
    output_path = output_root / "external_reviews" / f"busi_{method_id}_frozen_review.json"
    if output_path.exists() and not force:
        with output_path.open("r", encoding="utf-8") as handle:
            return {"returncode": 0, "output_path": str(output_path), "cached": True, "report": json.load(handle)}
    command = [
        sys.executable,
        "scripts/eval_busi.py",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    review: dict[str, Any] = {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "output_path": str(output_path),
    }
    if completed.returncode == 0 and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            review["report"] = json.load(handle)
    return review


def _external_metrics(item: dict[str, Any]) -> dict[str, Any]:
    review = item.get("external_review")
    if not isinstance(review, dict):
        return {}
    report = review.get("report")
    if not isinstance(report, dict):
        return {}
    metrics = report.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _decision(item: dict[str, Any]) -> str:
    metrics = _external_metrics(item)
    auc = metrics.get("auc")
    if not isinstance(auc, (int, float)):
        return "report_only"
    if (
        float(auc) > MAINLINE_METRICS["auc"]
        and float(metrics.get("sensitivity", 0.0)) >= MAINLINE_METRICS["sensitivity"] - 0.01
        and float(metrics.get("specificity", 0.0)) >= MAINLINE_METRICS["specificity"] - 0.01
        and float(metrics.get("f1_score", 0.0)) >= MAINLINE_METRICS["f1_score"] - 0.01
    ):
        return "candidate_hold_for_user"
    return "report_only"


def _markdown(report: dict[str, Any]) -> list[str]:
    lines = [
        "# 非 SAM 分割模型替换主线 ROI 外部验证",
        "",
        "## 实验边界",
        "",
        "- 训练集：BUSBRA，用于分割器训练和内部分割指标记录。",
        "- 外部验证：BUSI，每个冻结配置只运行一次，不根据 BUSI 结果回调分割器或 ROI 参数。",
        "- 主线分类器、分类阈值、ROI stacker、ROI area gate 保持 `configs/inference/demo.yml` 原设置；本实验只替换分割器 checkpoint。",
        "- SAM、BUSSAM、Auto-BUSAM 等 prompt/adapter 路线未纳入本轮实验。",
        "",
        "## 主线对照",
        "",
        f"- BUSI AUC：`{MAINLINE_METRICS['auc']:.4f}`",
        f"- Threshold：`{MAINLINE_METRICS['threshold']:.2f}`",
        f"- Accuracy / Sensitivity / Specificity / F1：`{MAINLINE_METRICS['accuracy']:.4f}` / `{MAINLINE_METRICS['sensitivity']:.4f}` / `{MAINLINE_METRICS['specificity']:.4f}` / `{MAINLINE_METRICS['f1_score']:.4f}`",
        "",
        "## 结果总表",
        "",
        "| 方法 | 类别 | 状态 | Val Dice | Val IoU | Boundary F1 | HD95 | BUSI AUC | BUSI Acc | Sens | Spec | F1 | 决策 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["results"]:
        internal = item.get("internal_mean_metrics", {})
        external = _external_metrics(item)
        lines.append(
            "| "
            + " | ".join(
                [
                    item["method_id"],
                    item["category"],
                    item["status"],
                    f"{float(internal['dice']):.4f}" if "dice" in internal else "-",
                    f"{float(internal['iou']):.4f}" if "iou" in internal else "-",
                    f"{float(internal['boundary_f1']):.4f}" if "boundary_f1" in internal else "-",
                    f"{float(internal['hd95']):.2f}" if "hd95" in internal and np.isfinite(float(internal["hd95"])) else "-",
                    f"{float(external['auc']):.4f}" if "auc" in external else "-",
                    f"{float(external['accuracy']):.4f}" if "accuracy" in external else "-",
                    f"{float(external['sensitivity']):.4f}" if "sensitivity" in external else "-",
                    f"{float(external['specificity']):.4f}" if "specificity" in external else "-",
                    f"{float(external['f1_score']):.4f}" if "f1_score" in external else "-",
                    item.get("decision", "report_only"),
                ]
            )
            + " |"
        )
    best = report.get("best_by_auc")
    lines.extend(["", "## 结论", ""])
    if isinstance(best, dict):
        delta = float(best.get("busi_auc", 0.0)) - MAINLINE_METRICS["auc"]
        lines.append(
            f"- 本轮最佳替换分割器为 `{best['method_id']}`，BUSI AUC `{best['busi_auc']:.4f}`，相对主线变化 `{delta:+.4f}`。"
        )
    lines.extend(
        [
            "- 若 `决策` 为 `report_only`，表示该方法未达到“外部 AUC 高于主线且关键指标无明显下降”的候选保留条件。",
            "- 本报告不自动修改 `configs/inference/demo.yml`，也不写入 README；是否合入需要人工选择。",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_config = _load_yaml(args.base_config)
    _, paths = load_project_config(args.base_config)
    output_root = (paths.project_root / REPORT_DIR).resolve()
    generated_config_root = output_root / "generated_configs"
    output_root.mkdir(parents=True, exist_ok=True)
    generated_config_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for method in _selected_methods(args.methods):
        item: dict[str, Any] = {
            "method_id": method.method_id,
            "category": method.category,
            "rationale": method.rationale,
            "status": "pending",
            "decision": "report_only",
        }
        config = _method_config(
            base_config=base_config,
            method=method,
            output_root=output_root,
            fold_count=int(args.fold_count),
            device=args.device,
        )
        config_path = _write_yaml(generated_config_root / f"{method.method_id}.yml", config)
        item["config_path"] = str(config_path)
        try:
            fold_reports: list[dict[str, Any]] = []
            for fold in range(1, int(args.fold_count) + 1):
                cached = None if args.force else _load_cached_fold_report(paths, config, fold)
                if cached is not None:
                    fold_reports.append(cached)
                else:
                    fold_reports.append(run_segmentation_training(config_path, fold=fold, epochs_override=int(args.epochs)))
            item["fold_reports"] = fold_reports
            item["internal_mean_metrics"] = _mean_metrics(fold_reports)
            item["frozen_config"] = str(
                _freeze_runtime_config(
                    runtime_config_path=args.runtime_config,
                    method=method,
                    fold_reports=fold_reports,
                    output_root=output_root,
                )
            )
            item["status"] = "trained"
            if not args.no_busi:
                item["external_review"] = _run_busi_review(
                    Path(item["frozen_config"]),
                    output_root,
                    method.method_id,
                    force=bool(args.force),
                )
                item["decision"] = _decision(item)
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)

    best_item = None
    best_auc = -1.0
    for item in results:
        metrics = _external_metrics(item)
        auc = metrics.get("auc")
        if isinstance(auc, (int, float)) and float(auc) > best_auc:
            best_auc = float(auc)
            best_item = {"method_id": item["method_id"], "busi_auc": best_auc}
    report = {
        "generated_at": timestamp_now(),
        "base_config": args.base_config,
        "runtime_config": args.runtime_config,
        "fold_count": int(args.fold_count),
        "epochs": int(args.epochs),
        "mainline_metrics": MAINLINE_METRICS,
        "data_boundary": "BUSBRA-only segmenter training; BUSI frozen external review only.",
        "methods": [method.method_id for method in _selected_methods(args.methods)],
        "results": results,
        "best_by_auc": best_item,
    }
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, _markdown(report))
    print(
        {
            "methods": len(results),
            "trained": sum(1 for item in results if item["status"] == "trained"),
            "failed": sum(1 for item in results if item["status"] == "failed"),
            "best_by_auc": best_item,
            "markdown": args.markdown,
        }
    )
    return report


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
