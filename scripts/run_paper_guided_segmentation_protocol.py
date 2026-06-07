"""Utility script for paper guided segmentation protocol workflows."""

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


REPORT_DIR = Path("artifacts/reports/Chinese reports/06_paper_guided_optimization")
MAINLINE_BUSI_AUC = 0.9255802549852892
MAINLINE_METRICS = {
    "auc": MAINLINE_BUSI_AUC,
    "threshold": 0.51,
    "sensitivity": 0.8666666666666667,
    "specificity": 0.8466819221967964,
    "accuracy": 0.8531684698608965,
    "precision": 0.7309236947791165,
    "f1_score": 0.7930283224400873,
}


@dataclass(frozen=True)
class MethodSpec:
    """Represent MethodSpec for this module."""
    method_id: str
    paper_source: str
    rationale: str
    model_overrides: dict[str, Any]
    loss: dict[str, Any]


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        method_id="baseline_unet_bce",
        paper_source="control",
        rationale="原始 U-Net + BCE 对照，用于确认新训练协议没有改变数据划分。",
        model_overrides={"architecture": "unet", "encoder_name": "resnet18", "encoder_weights": "imagenet"},
        loss={"name": "bce", "bce_weight": 1.0},
    ),
    MethodSpec(
        method_id="baseline_unet_bce_dice",
        paper_source="segmentation loss ablation",
        rationale="BCE 提供像素级校准，Dice 约束前景重叠，用作后续边界/上下文模块的强基线。",
        model_overrides={"architecture": "unet", "encoder_name": "resnet18", "encoder_weights": "imagenet"},
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    MethodSpec(
        method_id="cenet_dseb_only",
        paper_source="CENet DSEB",
        rationale="只启用多尺度差分边缘增强和差分注意力，检验边界跳连是否改善 BUS 病灶 mask。",
        model_overrides={
            "architecture": "cenet_lite",
            "base_channels": 24,
            "use_dseb": True,
            "use_cfam": False,
            "use_nonlocal": False,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    MethodSpec(
        method_id="cenet_cfam_only",
        paper_source="CENet CFAM",
        rationale="只启用通道校准、多尺度上下文聚合和轻量 non-local，检验上下文建模收益。",
        model_overrides={
            "architecture": "cenet_lite",
            "base_channels": 24,
            "use_dseb": False,
            "use_cfam": True,
            "use_nonlocal": True,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    MethodSpec(
        method_id="cenet_dseb_cfam",
        paper_source="CENet DSEB + CFAM",
        rationale="同时加入边缘增强和上下文融合，作为 CENet-lite 主实验。",
        model_overrides={
            "architecture": "cenet_lite",
            "base_channels": 24,
            "use_dseb": True,
            "use_cfam": True,
            "use_nonlocal": True,
        },
        loss={"name": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0},
    ),
    MethodSpec(
        method_id="cenet_boundary_head",
        paper_source="CENet boundary enhancement",
        rationale="在 CENet-lite 上增加边界辅助头，约束模糊边界和小结构。",
        model_overrides={
            "architecture": "cenet_lite",
            "base_channels": 24,
            "use_dseb": True,
            "use_cfam": True,
            "use_nonlocal": True,
            "boundary_head": True,
        },
        loss={"name": "bce_dice_boundary", "bce_weight": 1.0, "dice_weight": 1.0, "boundary_weight": 0.25},
    ),
    MethodSpec(
        method_id="cscpa_pal_lite",
        paper_source="CSC-PA PAL",
        rationale="用像素邻域亲和一致性约束边界局部结构，是 labeled-only 的 PAL 近似。",
        model_overrides={
            "architecture": "cenet_lite",
            "base_channels": 24,
            "use_dseb": True,
            "use_cfam": True,
            "use_nonlocal": True,
            "boundary_head": True,
        },
        loss={
            "name": "bce_dice_boundary_pal",
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "boundary_weight": 0.25,
            "pal_weight": 0.15,
        },
    ),
    MethodSpec(
        method_id="cscpa_foreground_edge_proto",
        paper_source="CSC-PA FPA + EPA",
        rationale="在无额外无标注数据时，用 BUSBRA 训练折内前景/边界原型对比损失近似 FPA/EPA。",
        model_overrides={
            "architecture": "cenet_lite",
            "base_channels": 24,
            "use_dseb": True,
            "use_cfam": True,
            "use_nonlocal": True,
            "boundary_head": True,
        },
        loss={
            "name": "bce_dice_boundary_pal_prototype",
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "boundary_weight": 0.25,
            "pal_weight": 0.15,
            "foreground_prototype_weight": 0.08,
            "edge_prototype_weight": 0.05,
        },
    ),
    MethodSpec(
        method_id="cenet_pvtv2_boundary_pal",
        paper_source="CENet PvT-v2 encoder",
        rationale="尝试使用 PVT-v2 特征金字塔编码器复现论文主干方向；若 timm 不支持则记录依赖受限。",
        model_overrides={
            "architecture": "cenet_pvtv2",
            "encoder_name": "pvt_v2_b0",
            "encoder_weights": None,
            "decoder_channels": 96,
            "use_dseb": True,
            "use_cfam": True,
            "use_nonlocal": True,
            "boundary_head": True,
        },
        loss={
            "name": "bce_dice_boundary_pal",
            "bce_weight": 1.0,
            "dice_weight": 1.0,
            "boundary_weight": 0.25,
            "pal_weight": 0.15,
        },
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-guided segmentation experiments under the BUSBRA-only selection protocol. "
            "BUSI review is optional and is only executed for frozen successful candidates."
        )
    )
    parser.add_argument("--base-config", default="configs/segmenter/unet_5fold.yml")
    parser.add_argument("--runtime-config", default="configs/inference/demo.yml")
    parser.add_argument("--stage", choices=("smoke", "screening", "full"), default="smoke")
    parser.add_argument("--methods", default="all", help="Comma-separated method ids, or all.")
    parser.add_argument("--fold-count", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--run-busi", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(REPORT_DIR / "paper_guided_segmentation_protocol.json"))
    parser.add_argument("--markdown", default=str(REPORT_DIR / "paper_guided_segmentation_protocol.md"))
    return parser


def _selected_methods(value: str) -> list[MethodSpec]:
    """Resolve selected experiment method definitions."""
    if value == "all":
        return list(METHODS)
    selected = {item.strip() for item in value.split(",") if item.strip()}
    method_by_id = {method.method_id: method for method in METHODS}
    missing = sorted(selected - set(method_by_id))
    if missing:
        raise ValueError(f"Unknown method ids: {missing}")
    return [method_by_id[method_id] for method_id in selected]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping used to materialize experiment configs."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write a YAML experiment config with stable key ordering."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def _experiment_config(
    *,
    base_config: dict[str, Any],
    method: MethodSpec,
    stage: str,
    fold_count: int,
    output_root: Path,
    device: str | None,
) -> dict[str, Any]:
    """Build one experiment config from base settings."""
    config = copy.deepcopy(base_config)
    config.setdefault("model", {})
    config["model"].update(method.model_overrides)
    config["loss"] = copy.deepcopy(method.loss)
    config["paper_guided_experiment"] = {
        "method_id": method.method_id,
        "paper_source": method.paper_source,
        "rationale": method.rationale,
            "stage": stage,
        "generated_at": timestamp_now(),
        "data_boundary": "BUSBRA only for training/model selection; BUSI external review is frozen and non-tuning.",
    }
    if device:
        config["device"] = device
    config.setdefault("training", {})
    config["training"]["fold_count"] = int(fold_count)
    config.setdefault("output", {})
    config["output"]["checkpoint_name"] = f"paper_guided_{method.method_id}_fold{{fold}}.pt"
    config["output"]["report_name"] = f"paper_guided_{method.method_id}_train_fold{{fold}}.json"
    config.setdefault("data", {})
    if stage == "smoke":
        config["data"]["batch_size"] = min(int(config["data"].get("batch_size", 4)), 2)
        config["training"]["max_train_batches"] = 2
        config["training"]["max_val_batches"] = 2
    config["report_dir"] = str(output_root)
    return config


def _mean_metrics(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Average metric dictionaries across completed folds."""
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
    method: MethodSpec,
    fold_reports: list[dict[str, Any]],
    output_root: Path,
) -> Path:
    """Freeze runtime config."""
    runtime = _load_yaml(runtime_config_path)
    checkpoints = [report["checkpoint_path"] for report in sorted(fold_reports, key=lambda item: item["fold"])]
    runtime.setdefault("runtime", {})
    runtime["runtime"]["segmenter_checkpoint"] = None
    runtime["runtime"]["segmenter_checkpoints"] = checkpoints
    runtime["runtime"]["ensemble_display_name"] = (
        f"{runtime['runtime'].get('ensemble_display_name', 'BUCAD')} + {method.method_id}"
    )
    runtime["paper_guided_experiment"] = {
        "method_id": method.method_id,
        "paper_source": method.paper_source,
        "frozen_at": timestamp_now(),
        "note": "Frozen for one-pass external BUSI review; do not tune parameters from BUSI output.",
    }
    output_path = output_root / "frozen_configs" / f"demo_{method.method_id}.yml"
    _write_yaml(output_path, runtime)
    return output_path


def _run_busi_review(config_path: Path, output_root: Path, method_id: str, *, stage: str) -> dict[str, Any]:
    """Run busi review."""
    output_path = output_root / "external_reviews" / f"busi_{stage}_{method_id}_frozen_review.json"
    command = [
        sys.executable,
        "scripts/eval_busi.py",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    result: dict[str, Any] = {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "output_path": str(output_path),
    }
    if completed.returncode == 0 and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            result["report"] = json.load(handle)
    return result


def _candidate_decision(external_review: dict[str, Any] | None) -> str:
    """Summarize whether a candidate meets the promotion criteria."""
    if not external_review or external_review.get("returncode") != 0:
        return "report_only"
    report = external_review.get("report", {})
    metrics = report.get("metrics") or report.get("fixed_threshold_metrics") or {}
    auc = metrics.get("auc")
    sensitivity = metrics.get("sensitivity")
    specificity = metrics.get("specificity")
    f1_score = metrics.get("f1_score")
    if not isinstance(auc, (int, float)):
        return "report_only"
    if (
        float(auc) > MAINLINE_METRICS["auc"]
        and float(sensitivity or 0.0) >= MAINLINE_METRICS["sensitivity"] - 0.01
        and float(specificity or 0.0) >= MAINLINE_METRICS["specificity"] - 0.01
        and float(f1_score or 0.0) >= MAINLINE_METRICS["f1_score"] - 0.01
    ):
        return "candidate_hold_for_user"
    return "report_only"


def _build_markdown(report: dict[str, Any]) -> list[str]:
    """Build the Markdown report body for this experiment."""
    lines = [
        "# 论文启发分割与 ROI 优化实验记录",
        "",
        "## 实验边界",
        "",
        "- BUSBRA 用于训练、内部验证、OOF 和候选筛选。",
        "- BUSI 只用于冻结配置后的单次外部复核；不得根据 BUSI 结果回调参数。",
        "- 主线 `configs/inference/demo.yml` 不在本脚本中修改。",
        "",
        "## 主线对照",
        "",
        f"- BUSI AUC：`{MAINLINE_METRICS['auc']:.4f}`",
        f"- 阈值：`{MAINLINE_METRICS['threshold']:.2f}`",
        f"- Sensitivity / Specificity / F1：`{MAINLINE_METRICS['sensitivity']:.4f}` / `{MAINLINE_METRICS['specificity']:.4f}` / `{MAINLINE_METRICS['f1_score']:.4f}`",
        "",
        "## 方法结果",
        "",
        "| 方法 | 来源 | 状态 | 内部Dice | 内部IoU | Boundary F1 | HD95 | BUSI AUC | 决策 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["results"]:
        mean_metrics = item.get("internal_mean_metrics", {})
        external = item.get("external_review") or {}
        external_report = external.get("report", {}) if isinstance(external, dict) else {}
        metrics = external_report.get("metrics") or external_report.get("fixed_threshold_metrics") or {}
        busi_auc = metrics.get("auc")
        lines.append(
            "| "
            + " | ".join(
                [
                    item["method_id"],
                    item["paper_source"],
                    item["status"],
                    f"{mean_metrics.get('dice', 0.0):.4f}" if "dice" in mean_metrics else "-",
                    f"{mean_metrics.get('iou', 0.0):.4f}" if "iou" in mean_metrics else "-",
                    f"{mean_metrics.get('boundary_f1', 0.0):.4f}" if "boundary_f1" in mean_metrics else "-",
                    f"{mean_metrics.get('hd95', 0.0):.2f}" if "hd95" in mean_metrics else "-",
                    f"{float(busi_auc):.4f}" if isinstance(busi_auc, (int, float)) else "-",
                    item.get("decision", "report_only"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `cenet_pvtv2_boundary_pal` 依赖 timm 对 PVT-v2 `features_only` 的支持；若构建失败，报告会记录为依赖受限。",
            "- `cscpa_foreground_edge_proto` 是 labeled-only 近似，不等同于 CSC-PA 原文的半监督跨图像原型注意力完整复现。",
            "- 决策为 `candidate_hold_for_user` 的配置只表示可供人工复核，不会自动合入主线。",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the experiment workflow and return the generated summary."""
    base_config = _load_yaml(args.base_config)
    _, paths = load_project_config(args.base_config)
    output_root = (paths.project_root / REPORT_DIR).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    generated_config_root = output_root / "generated_configs"
    generated_config_root.mkdir(parents=True, exist_ok=True)

    methods = _selected_methods(str(args.methods))
    default_fold_count = 1 if args.stage in {"smoke", "screening"} else base_config.get("training", {}).get("fold_count", 5)
    default_epochs = (
        args.smoke_epochs
        if args.stage == "smoke"
        else 1
        if args.stage == "screening"
        else base_config.get("training", {}).get("epochs", 5)
    )
    fold_count = int(args.fold_count or default_fold_count)
    epochs = int(args.epochs or default_epochs)

    results = []
    for method in methods:
        item: dict[str, Any] = {
            "method_id": method.method_id,
            "paper_source": method.paper_source,
            "rationale": method.rationale,
            "status": "pending",
            "decision": "report_only",
        }
        config = _experiment_config(
            base_config=base_config,
            method=method,
            stage=args.stage,
            fold_count=fold_count,
            output_root=output_root,
            device=args.device,
        )
        config_path = generated_config_root / f"{method.method_id}.yml"
        _write_yaml(config_path, config)
        item["config_path"] = str(config_path)
        fold_reports: list[dict[str, Any]] = []
        try:
            for fold in range(1, fold_count + 1):
                fold_reports.append(run_segmentation_training(config_path, fold=fold, epochs_override=epochs))
            item["status"] = "trained"
            item["fold_reports"] = fold_reports
            item["internal_mean_metrics"] = _mean_metrics(fold_reports)
            frozen_config = _freeze_runtime_config(
                runtime_config_path=args.runtime_config,
                method=method,
                fold_reports=fold_reports,
                output_root=output_root,
            )
            item["frozen_config"] = str(frozen_config)
            if args.run_busi:
                item["external_review"] = _run_busi_review(
                    frozen_config,
                    output_root,
                    method.method_id,
                    stage=args.stage,
                )
                item["decision"] = _candidate_decision(item["external_review"])
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)

    report = {
        "generated_at": timestamp_now(),
        "stage": args.stage,
        "fold_count": fold_count,
        "epochs": epochs,
        "mainline_metrics": MAINLINE_METRICS,
        "data_boundary": "BUSBRA-only training and selection; BUSI frozen one-pass review only.",
        "results": results,
    }
    write_json_report(args.output, report)
    write_markdown_report(args.markdown, _build_markdown(report))
    return report


def main() -> int:
    """Parse CLI arguments and run the script entry point."""
    args = build_parser().parse_args()
    report = run(args)
    print(
        {
            "stage": report["stage"],
            "methods": len(report["results"]),
            "trained": sum(1 for item in report["results"] if item["status"] == "trained"),
            "failed": sum(1 for item in report["results"] if item["status"] == "failed"),
            "markdown": args.markdown,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
