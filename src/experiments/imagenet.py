"""ImageNet-1K preparation and fair evaluation helpers for CNN architecture studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from src.engine.checkpoints import atomic_torch_save
from src.models.classifier import create_classifier
from src.utils.config import load_project_config
from src.utils.reporting import write_json_report, write_markdown_report
from src.utils.runtime import optional_import, require_dependency, seed_everything, select_device


torch = optional_import("torch")
timm_loss = optional_import("timm.loss")
timm_data = optional_import("timm.data")
timm_utils = optional_import("timm.utils")
torchvision_datasets = optional_import("torchvision.datasets")
torchvision_imagenet = optional_import("torchvision.datasets.imagenet")
torchvision_transforms = optional_import("torchvision.transforms")
torch_utils_data = optional_import("torch.utils.data")
torch_optim = optional_import("torch.optim")


ARCHIVE_META = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
}


def build_prepare_parser() -> argparse.ArgumentParser:
    """Build parser for ImageNet archive validation and preparation."""
    parser = argparse.ArgumentParser(description="Prepare authorized ImageNet-1K archives.")
    parser.add_argument("--config", default="configs/imagenet/sonoglore_convnext_v1_eval.yml")
    parser.add_argument("--root", default=None, help="ImageNet root containing ILSVRC2012 archives.")
    parser.add_argument("--skip-md5", action="store_true", help="Skip archive MD5 checks.")
    parser.add_argument("--prepare-train", action="store_true", help="Prepare the train split.")
    parser.add_argument("--prepare-val", action="store_true", help="Prepare the val split.")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be prepared.")
    return parser


def build_eval_parser() -> argparse.ArgumentParser:
    """Build parser for one ImageNet model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate one configured model on ImageNet-1K.")
    parser.add_argument("--config", default="configs/imagenet/sonoglore_convnext_v1_eval.yml")
    parser.add_argument("--root", default=None, help="Override ImageNet root.")
    parser.add_argument("--split", default=None, choices=["train", "val"], help="Override split.")
    parser.add_argument("--output", default=None, help="Override JSON report path.")
    parser.add_argument("--checkpoint", default=None, help="Override model checkpoint path.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for smoke tests.")
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    """Build parser for ImageNet model training."""
    parser = argparse.ArgumentParser(description="Train one configured model on ImageNet-1K.")
    parser.add_argument("--config", default="configs/imagenet/sonoglore_convnext_v1_train.yml")
    parser.add_argument("--root", default=None, help="Override ImageNet root.")
    parser.add_argument("--output", default=None, help="Override JSON report path.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint output path.")
    parser.add_argument("--epochs-override", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit train batches per epoch for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit validation batches for smoke tests.")
    parser.add_argument("--skip-val", action="store_true", help="Skip validation after each epoch.")
    return parser


def build_ablation_parser() -> argparse.ArgumentParser:
    """Build parser for multi-model ImageNet ablation evaluation."""
    parser = argparse.ArgumentParser(description="Run ImageNet model ablations.")
    parser.add_argument("--config", default="configs/imagenet/sonoglore_convnext_v1_ablation.yml")
    parser.add_argument("--root", default=None, help="Override ImageNet root.")
    parser.add_argument("--output", default=None, help="Override JSON report path.")
    parser.add_argument("--markdown", default=None, help="Override Markdown report path.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for smoke tests.")
    return parser


def prepare_imagenet_archives(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
    skip_md5: bool = False,
    prepare_train: bool = False,
    prepare_val: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate official ImageNet archives and optionally let torchvision parse them."""
    require_dependency("torchvision.datasets", torchvision_datasets)
    require_dependency("torchvision.datasets.imagenet", torchvision_imagenet)
    config, paths = load_project_config(config_path)
    root_path = _resolve_imagenet_root(config, paths, root)
    archive_status = {
        key: _archive_status(root_path, name, expected_md5, skip_md5=skip_md5)
        for key, (name, expected_md5) in ARCHIVE_META.items()
    }
    requested_splits = []
    if prepare_train:
        requested_splits.append("train")
    if prepare_val:
        requested_splits.append("val")
    if not requested_splits:
        requested_splits.append(str(config.get("dataset", {}).get("split", "val")))
    prepared = []
    errors = []
    for split in requested_splits:
        split = str(split).lower()
        if split not in {"train", "val"}:
            errors.append(f"Unsupported ImageNet split: {split}")
            continue
        required = ["devkit", split]
        missing = [key for key in required if not archive_status[key]["exists"]]
        invalid = [key for key in required if not archive_status[key]["md5_ok"]]
        if missing or invalid:
            errors.append(
                f"Cannot prepare {split}: missing={missing or []}, md5_failed={invalid or []}."
            )
            continue
        if dry_run:
            prepared.append({"split": split, "dry_run": True})
            continue
        try:
            dataset = torchvision_datasets.ImageNet(str(root_path), split=split)
            prepared.append({"split": split, "samples": len(dataset), "dry_run": False})
        except Exception as exc:
            errors.append(f"torchvision failed to prepare {split}: {exc}")
    return {
        "root": str(root_path),
        "archives": archive_status,
        "requested_splits": requested_splits,
        "prepared": prepared,
        "errors": errors,
        "instructions": [
            "Download ImageNet-1K ILSVRC2012 archives from the official ImageNet site after accepting the terms.",
            "Place ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar, and ILSVRC2012_devkit_t12.tar.gz in the configured root.",
            "Run scripts/prepare_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_eval.yml --prepare-val.",
        ],
    }


def evaluate_imagenet_config(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
    split: str | None = None,
    output: str | Path | None = None,
    checkpoint: str | Path | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate one configured model on an ImageNet split and write a JSON report."""
    require_dependency("torch", torch)
    config, paths = load_project_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device = select_device(str(config.get("device", "auto")))
    root_path = _resolve_imagenet_root(config, paths, root)
    split_name = str(split or config.get("dataset", {}).get("split", "val")).lower()
    _require_prepared_imagenet(root_path, split_name)
    model_cfg = dict(config.get("model", {}))
    checkpoint_path = _resolve_checkpoint_path(config, paths, checkpoint)
    if checkpoint_path is None and _requires_trained_checkpoint(model_cfg):
        raise ValueError(
            f"{model_cfg.get('name')} requires a full ImageNet-trained checkpoint for fair evaluation. "
            "Train it with scripts/train_imagenet.py or pass --checkpoint."
        )
    model = _build_model(model_cfg, device=device, checkpoint_path=checkpoint_path)
    loader = _build_imagenet_loader(
        config=config,
        model=model,
        root=root_path,
        split=split_name,
    )
    result = _evaluate_model(
        model,
        loader,
        device=device,
        amp=bool(config.get("evaluation", {}).get("amp", True)),
        channels_last=bool(config.get("evaluation", {}).get("channels_last", False)),
        max_batches=max_batches,
    )
    report = {
        "config_path": str(config_path),
        "root": str(root_path),
        "split": split_name,
        "model": model_cfg,
        "metrics": result,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "dataset_size": len(loader.dataset),
        "max_batches": max_batches,
        "parameter_count": _count_parameters(model),
        "trainable_parameter_count": _count_parameters(model, trainable_only=True),
        "device": device,
        "data_config": _resolve_data_config(model, config.get("data", {})),
    }
    output_path = Path(output) if output is not None else paths.reports_root / str(
        config.get("output", {}).get("report_name", "imagenet_eval.json")
    )
    write_json_report(output_path, report)
    report["report_path"] = str(output_path)
    return report


def train_imagenet_config(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
    output: str | Path | None = None,
    checkpoint: str | Path | None = None,
    epochs_override: int | None = None,
    max_batches: int | None = None,
    max_val_batches: int | None = None,
    skip_val: bool = False,
) -> dict[str, Any]:
    """Train one configured ImageNet model and write checkpoint/report artifacts."""
    require_dependency("torch", torch)
    require_dependency("torch.optim", torch_optim)
    config, paths = load_project_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device = select_device(str(config.get("device", "auto")))
    root_path = _resolve_imagenet_root(config, paths, root)
    _require_prepared_imagenet(root_path, "train")
    if not skip_val:
        _require_prepared_imagenet(root_path, "val")

    model_cfg = dict(config.get("model", {}))
    model = _build_model(model_cfg, device=device)
    data_cfg = dict(config.get("data", {}))
    train_loader = _build_imagenet_loader(
        config=config,
        model=model,
        root=root_path,
        split="train",
        is_training=True,
    )
    val_loader = None
    if not skip_val:
        val_loader = _build_imagenet_loader(
            config=config,
            model=model,
            root=root_path,
            split="val",
            is_training=False,
        )

    training_cfg = dict(config.get("training", {}))
    epochs = int(epochs_override or training_cfg.get("epochs", 300))
    base_lr = _scaled_learning_rate(training_cfg, batch_size=int(data_cfg.get("batch_size", 64)))
    optimizer = _build_optimizer(model, training_cfg, learning_rate=base_lr)
    mixup_fn = _build_mixup_fn(training_cfg, num_classes=int(config.get("dataset", {}).get("num_classes", 1000)))
    criterion = _build_training_criterion(training_cfg, mixup_enabled=mixup_fn is not None)
    ema_model = _build_ema_model(model, training_cfg, device=device)
    channels_last = bool(training_cfg.get("channels_last", config.get("evaluation", {}).get("channels_last", False)))

    best_top1 = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    epoch_reports: list[dict[str, Any]] = []
    for epoch in range(epochs):
        current_lr = _lr_for_epoch(
            epoch,
            base_lr=base_lr,
            epochs=epochs,
            warmup_epochs=int(training_cfg.get("warmup_epochs", 20)),
            min_lr=float(training_cfg.get("min_lr", 0.0)),
        )
        _set_optimizer_lr(optimizer, current_lr)
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            amp=bool(training_cfg.get("amp", True)),
            channels_last=channels_last,
            mixup_fn=mixup_fn,
            ema_model=ema_model,
            max_batches=max_batches,
        )
        val_metrics = None
        if val_loader is not None:
            eval_model = ema_model.module if ema_model is not None and bool(training_cfg.get("eval_ema", True)) else model
            val_metrics = _evaluate_model(
                eval_model,
                val_loader,
                device=device,
                amp=bool(config.get("evaluation", {}).get("amp", True)),
                channels_last=bool(config.get("evaluation", {}).get("channels_last", channels_last)),
                max_batches=max_val_batches,
            )
            top1 = float(val_metrics.get("top1", 0.0))
            if top1 > best_top1:
                best_top1 = top1
                best_epoch = epoch + 1
                best_metrics = dict(val_metrics)
        epoch_reports.append(
            {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

    checkpoint_path = _resolve_output_checkpoint_path(config, paths, checkpoint)
    checkpoint_model = ema_model.module if ema_model is not None and bool(training_cfg.get("save_ema", True)) else model
    atomic_torch_save(
        {
            "state_dict": checkpoint_model.state_dict(),
            "model_config": model_cfg,
            "data_config": data_cfg,
            "training_config": training_cfg,
            "dataset": config.get("dataset", {}),
            "epoch": epochs,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "input_benchmark": "ImageNet-1K",
        },
        checkpoint_path,
    )
    report = {
        "config_path": str(config_path),
        "root": str(root_path),
        "model": model_cfg,
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "epochs": epochs,
        "max_batches": max_batches,
        "max_val_batches": max_val_batches,
        "skip_val": bool(skip_val),
        "dataset_size": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset) if val_loader is not None else None,
        },
        "parameter_count": _count_parameters(model),
        "trainable_parameter_count": _count_parameters(model, trainable_only=True),
        "data_config": _resolve_data_config(model, data_cfg),
        "training": {
            **training_cfg,
            "resolved_learning_rate": base_lr,
            "effective_batch_size": int(data_cfg.get("batch_size", 64)),
        },
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "epoch_reports": epoch_reports,
    }
    output_path = Path(output) if output is not None else paths.reports_root / str(
        config.get("output", {}).get("report_name", "imagenet_train.json")
    )
    write_json_report(output_path, report)
    report["report_path"] = str(output_path)
    return report


def run_imagenet_ablation(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
    output: str | Path | None = None,
    markdown: str | Path | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate all models listed in one ImageNet ablation config."""
    require_dependency("torch", torch)
    config, paths = load_project_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    root_path = _resolve_imagenet_root(config, paths, root)
    split_name = str(config.get("dataset", {}).get("split", "val")).lower()
    _require_prepared_imagenet(root_path, split_name)
    rows = []
    dataset_size = None
    for entry in config.get("models", []):
        model_cfg = dict(entry.get("model", {}))
        checkpoint_path = _resolve_checkpoint_path(
            {"model": model_cfg, "evaluation": entry.get("evaluation", {})},
            paths,
            entry.get("checkpoint") or model_cfg.get("checkpoint"),
        )
        if checkpoint_path is not None and not checkpoint_path.exists():
            rows.append(
                {
                    "id": entry.get("id"),
                    "name": entry.get("name"),
                    "model": model_cfg,
                    "checkpoint_path": str(checkpoint_path),
                    "status": "requires_training",
                    "metrics": None,
                    "parameter_count": None,
                }
            )
            continue
        if checkpoint_path is None and _requires_trained_checkpoint(model_cfg):
            rows.append(
                {
                    "id": entry.get("id"),
                    "name": entry.get("name"),
                    "model": model_cfg,
                    "checkpoint_path": None,
                    "status": "requires_training",
                    "metrics": None,
                    "parameter_count": None,
                }
            )
            continue
        model = _build_model(
            model_cfg,
            device=select_device(str(config.get("device", "auto"))),
            checkpoint_path=checkpoint_path,
        )
        loader = _build_imagenet_loader(
            config=config,
            model=model,
            root=root_path,
            split=split_name,
        )
        dataset_size = len(loader.dataset)
        metrics = _evaluate_model(
            model,
            loader,
            device=select_device(str(config.get("device", "auto"))),
            amp=bool(config.get("evaluation", {}).get("amp", True)),
            channels_last=bool(config.get("evaluation", {}).get("channels_last", False)),
            max_batches=max_batches,
        )
        rows.append(
            {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "model": model_cfg,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "status": "evaluated",
                "metrics": metrics,
                "parameter_count": _count_parameters(model),
            }
        )
    report = {
        "config_path": str(config_path),
        "root": str(root_path),
        "split": split_name,
        "dataset_size": dataset_size,
        "max_batches": max_batches,
        "rows": rows,
    }
    output_path = Path(output) if output is not None else paths.reports_root / str(
        config.get("output", {}).get("report_name", "imagenet_ablation.json")
    )
    write_json_report(output_path, report)
    markdown_path = Path(markdown) if markdown is not None else paths.reports_root / str(
        config.get("output", {}).get("markdown_name", "imagenet_ablation.md")
    )
    write_markdown_report(markdown_path, _ablation_markdown(report))
    report["report_path"] = str(output_path)
    report["markdown_path"] = str(markdown_path)
    return report


def _resolve_imagenet_root(config: dict[str, Any], paths, root: str | Path | None) -> Path:
    """Resolve the ImageNet root from CLI, config, or shared paths."""
    configured = root or config.get("dataset", {}).get("root") or paths.imagenet_root
    candidate = Path(configured).expanduser()
    if candidate.is_absolute():
        return candidate
    return (paths.project_root / candidate).resolve()


def _require_prepared_imagenet(root: Path, split: str) -> None:
    """Raise a focused error when ImageNet official archives are not ready."""
    required_keys = ["devkit", str(split).lower()]
    missing = []
    for key in required_keys:
        archive_name = ARCHIVE_META[key][0]
        if not (root / archive_name).exists():
            missing.append(archive_name)
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"ImageNet split {split!r} is not prepared under {root}. Missing: {missing_text}. "
            "Download the official ILSVRC2012 archives after accepting ImageNet terms, then run "
            "`python scripts/prepare_imagenet.py --config configs/imagenet/sonoglore_convnext_v1_eval.yml "
            f"--prepare-{split}`."
        )


def _archive_status(
    root: Path,
    name: str,
    expected_md5: str,
    *,
    skip_md5: bool,
) -> dict[str, Any]:
    """Return existence and checksum status for one ImageNet archive."""
    path = root / name
    exists = path.exists()
    digest = None
    md5_ok = bool(skip_md5) if exists else False
    if exists and not skip_md5:
        digest = _md5(path)
        md5_ok = digest.lower() == expected_md5.lower()
    return {
        "path": str(path),
        "exists": exists,
        "expected_md5": expected_md5,
        "actual_md5": digest,
        "md5_ok": md5_ok,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute an MD5 digest for one archive."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint_path(config: dict[str, Any], paths, checkpoint: str | Path | None = None) -> Path | None:
    """Resolve an optional checkpoint path from CLI, config, or model entry."""
    candidate = checkpoint
    if candidate is None:
        candidate = config.get("checkpoint")
    if candidate is None:
        candidate = config.get("evaluation", {}).get("checkpoint")
    if candidate is None:
        candidate = config.get("model", {}).get("checkpoint")
    if candidate in {None, ""}:
        return None
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path
    return (paths.project_root / path).resolve()


def _resolve_output_checkpoint_path(
    config: dict[str, Any],
    paths,
    checkpoint: str | Path | None,
) -> Path:
    """Resolve the training checkpoint destination."""
    candidate = checkpoint or config.get("output", {}).get("checkpoint_name", "imagenet_model.pt")
    path = Path(str(candidate)).expanduser()
    if path.is_absolute():
        return path
    return paths.checkpoints_root / path


def _load_state_dict(model, checkpoint_path: Path) -> dict[str, Any]:
    """Load a model checkpoint into the configured architecture."""
    require_dependency("torch", torch)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ImageNet checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("state_dict", state.get("model", state)) if isinstance(state, dict) else state
    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
    }


def _requires_trained_checkpoint(model_cfg: dict[str, Any]) -> bool:
    """Return True when pretrained backbone weights alone are not a fair ImageNet result."""
    name = str(model_cfg.get("name", "")).lower()
    return name.startswith("sonoglore_") or name.startswith("roi_dualview_")


def _build_model(model_cfg: dict[str, Any], *, device: str, checkpoint_path: Path | None = None):
    """Construct a configured classifier for ImageNet."""
    require_dependency("torch", torch)
    cfg = dict(model_cfg)
    model_name = str(cfg.pop("name", "convnext_tiny"))
    cfg.pop("checkpoint", None)
    pretrained = bool(cfg.pop("pretrained", True))
    in_chans = int(cfg.pop("in_chans", 3))
    num_classes = int(cfg.pop("num_classes", 1000))
    model = create_classifier(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=num_classes,
        **cfg,
    )
    if checkpoint_path is not None:
        _load_state_dict(model, checkpoint_path)
    return model.to(device)


def _resolve_data_config(model, data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve timm preprocessing settings for one model."""
    preprocess = data_cfg.get("preprocess", {}) if isinstance(data_cfg, dict) else {}
    use_timm = bool(preprocess.get("use_timm_data_config", True))
    if use_timm and timm_data is not None:
        resolved = timm_data.resolve_model_data_config(model)
    else:
        resolved = {}
    image_size = int(data_cfg.get("image_size", resolved.get("input_size", (3, 224, 224))[-1]))
    input_size = tuple(resolved.get("input_size", (3, image_size, image_size)))
    input_size = (int(input_size[0]), int(input_size[1]), int(input_size[2]))
    mean = preprocess.get("mean")
    if mean is None:
        mean = resolved.get("mean", (0.485, 0.456, 0.406))
    std = preprocess.get("std")
    if std is None:
        std = resolved.get("std", (0.229, 0.224, 0.225))
    interpolation = preprocess.get("interpolation")
    if interpolation is None:
        interpolation = resolved.get("interpolation", "bicubic")
    crop_pct = preprocess.get("crop_pct")
    if crop_pct is None:
        crop_pct = resolved.get("crop_pct", 0.875)
    return {
        "input_size": input_size,
        "mean": tuple(float(value) for value in mean),
        "std": tuple(float(value) for value in std),
        "interpolation": str(interpolation),
        "crop_pct": float(crop_pct),
    }


def _build_imagenet_loader(
    *,
    config: dict[str, Any],
    model,
    root: Path,
    split: str,
    is_training: bool = False,
):
    """Build ImageNet dataset and DataLoader from the configured preprocessing."""
    require_dependency("torchvision.datasets", torchvision_datasets)
    require_dependency("torchvision.transforms", torchvision_transforms)
    require_dependency("torch.utils.data", torch_utils_data)
    data_cfg = config.get("data", {})
    resolved = _resolve_data_config(model, data_cfg)
    image_size = int(data_cfg.get("image_size", resolved["input_size"][-1]))
    transform = _build_transform(data_cfg, resolved, image_size=image_size, is_training=is_training)
    dataset = torchvision_datasets.ImageNet(str(root), split=split, transform=transform)
    return torch_utils_data.DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=bool(is_training),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=select_device(str(config.get("device", "auto"))).startswith("cuda"),
        persistent_workers=int(data_cfg.get("num_workers", 4)) > 0,
        drop_last=bool(is_training and data_cfg.get("drop_last", True)),
    )


def _build_transform(
    data_cfg: dict[str, Any],
    resolved: dict[str, Any],
    *,
    image_size: int,
    is_training: bool,
):
    """Build train/eval transforms with timm when available."""
    preprocess = data_cfg.get("preprocess", {}) if isinstance(data_cfg, dict) else {}
    if is_training and timm_data is not None:
        train_cfg = data_cfg.get("train_augmentation", {}) or {}
        return timm_data.create_transform(
            input_size=resolved["input_size"],
            is_training=True,
            scale=tuple(train_cfg.get("scale", (0.08, 1.0))),
            ratio=tuple(train_cfg.get("ratio", (3.0 / 4.0, 4.0 / 3.0))),
            hflip=float(train_cfg.get("hflip", 0.5)),
            vflip=float(train_cfg.get("vflip", 0.0)),
            color_jitter=train_cfg.get("color_jitter", 0.4),
            auto_augment=train_cfg.get("auto_augment", "rand-m9-mstd0.5-inc1"),
            interpolation=str(resolved.get("interpolation", "bicubic")),
            mean=resolved["mean"],
            std=resolved["std"],
            re_prob=float(train_cfg.get("random_erasing_prob", 0.25)),
            re_mode=str(train_cfg.get("random_erasing_mode", "pixel")),
            re_count=int(train_cfg.get("random_erasing_count", 1)),
        )
    crop_pct = float(resolved.get("crop_pct", 0.875))
    resize_size = max(image_size, int(round(image_size / crop_pct)))
    interpolation = _pil_interpolation(str(resolved.get("interpolation", "bicubic")))
    return torchvision_transforms.Compose(
        [
            torchvision_transforms.Resize(resize_size, interpolation=interpolation),
            torchvision_transforms.CenterCrop(image_size),
            torchvision_transforms.ToTensor(),
            torchvision_transforms.Normalize(resolved["mean"], resolved["std"]),
        ]
    )


def _pil_interpolation(name: str):
    """Map timm interpolation names to torchvision interpolation modes."""
    mode = getattr(torchvision_transforms, "InterpolationMode", None)
    if mode is None:
        return 3
    normalized = name.lower()
    if normalized in {"bicubic", "cubic"}:
        return mode.BICUBIC
    if normalized in {"bilinear", "linear"}:
        return mode.BILINEAR
    if normalized == "nearest":
        return mode.NEAREST
    return mode.BICUBIC


def _evaluate_model(
    model,
    loader,
    *,
    device: str,
    amp: bool,
    channels_last: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    """Return Top-1/Top-5 and throughput metrics for one ImageNet loader."""
    require_dependency("torch", torch)
    model.eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    correct1 = 0
    correct5 = 0
    total = 0
    forward_seconds = 0.0
    autocast_enabled = bool(amp and str(device).startswith("cuda"))
    context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with context():
        for batch_index, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if channels_last:
                images = images.to(memory_format=torch.channels_last)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            with _autocast_context(device, enabled=autocast_enabled):
                logits = model(images)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - start
            max_k = min(5, int(logits.shape[1]))
            _, pred = logits.topk(max_k, dim=1)
            pred = pred.t()
            correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
            correct1 += int(correct[:1].reshape(-1).float().sum().item())
            correct5 += int(correct[:max_k].reshape(-1).float().sum().item())
            total += int(targets.numel())
    top1 = correct1 / total if total else 0.0
    top5 = correct5 / total if total else 0.0
    images_per_second = total / forward_seconds if forward_seconds > 0 else 0.0
    return {
        "samples": total,
        "top1": top1,
        "top5": top5,
        "top1_error": 1.0 - top1 if total else 0.0,
        "top5_error": 1.0 - top5 if total else 0.0,
        "forward_seconds": forward_seconds,
        "images_per_second": images_per_second,
    }


def _scaled_learning_rate(training_cfg: dict[str, Any], *, batch_size: int) -> float:
    """Scale base LR by batch size when requested by the config."""
    base_lr = float(training_cfg.get("learning_rate", training_cfg.get("base_learning_rate", 4e-3)))
    reference_batch_size = int(training_cfg.get("reference_batch_size", 4096))
    if bool(training_cfg.get("scale_lr_by_batch_size", True)):
        return base_lr * float(batch_size) / float(max(1, reference_batch_size))
    return base_lr


def _build_optimizer(model, training_cfg: dict[str, Any], *, learning_rate: float):
    """Build the ImageNet optimizer."""
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported ImageNet optimizer: {optimizer_name}")
    return torch_optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        betas=tuple(training_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(training_cfg.get("weight_decay", 0.05)),
    )


def _build_mixup_fn(training_cfg: dict[str, Any], *, num_classes: int):
    """Build timm Mixup/CutMix when configured."""
    mixup_alpha = float(training_cfg.get("mixup_alpha", 0.8))
    cutmix_alpha = float(training_cfg.get("cutmix_alpha", 1.0))
    if max(mixup_alpha, cutmix_alpha) <= 0.0:
        return None
    require_dependency("timm.data", timm_data)
    return timm_data.Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=float(training_cfg.get("mix_probability", 1.0)),
        switch_prob=float(training_cfg.get("mix_switch_probability", 0.5)),
        label_smoothing=float(training_cfg.get("label_smoothing", 0.1)),
        num_classes=int(num_classes),
    )


def _build_training_criterion(training_cfg: dict[str, Any], *, mixup_enabled: bool):
    """Build ImageNet classification loss."""
    require_dependency("torch", torch)
    if mixup_enabled:
        require_dependency("timm.loss", timm_loss)
        return timm_loss.SoftTargetCrossEntropy()
    label_smoothing = float(training_cfg.get("label_smoothing", 0.1))
    if label_smoothing > 0.0 and timm_loss is not None:
        return timm_loss.LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    return torch.nn.CrossEntropyLoss()


def _build_ema_model(model, training_cfg: dict[str, Any], *, device: str):
    """Build optional EMA wrapper."""
    if not bool(training_cfg.get("use_ema", False)):
        return None
    require_dependency("timm.utils", timm_utils)
    ema_device = None if bool(training_cfg.get("ema_on_device", True)) else "cpu"
    if ema_device is None and not str(device).startswith("cuda"):
        ema_device = None
    return timm_utils.ModelEmaV2(
        model,
        decay=float(training_cfg.get("ema_decay", 0.9999)),
        device=ema_device,
    )


def _lr_for_epoch(
    epoch_index: int,
    *,
    base_lr: float,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
) -> float:
    """Return linear-warmup cosine-decay LR for one epoch."""
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        return base_lr * float(epoch_index + 1) / float(warmup_epochs)
    cosine_epochs = max(1, int(epochs) - int(warmup_epochs))
    cosine_index = min(max(epoch_index - int(warmup_epochs), 0), cosine_epochs)
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * float(cosine_index) / float(cosine_epochs)))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine_scale


def _set_optimizer_lr(optimizer, learning_rate: float) -> None:
    """Set all optimizer parameter groups to one LR."""
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def _train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    *,
    device: str,
    amp: bool,
    channels_last: bool,
    mixup_fn,
    ema_model,
    max_batches: int | None,
) -> dict[str, Any]:
    """Train one ImageNet epoch and return loss/throughput metrics."""
    require_dependency("torch", torch)
    model.train()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    total_loss = 0.0
    total_samples = 0
    forward_seconds = 0.0
    autocast_enabled = bool(amp and str(device).startswith("cuda"))
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        optimizer.zero_grad(set_to_none=True)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with _autocast_context(device, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            ema_model.update(model)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - start
        batch_size = int(images.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
    return {
        "samples": total_samples,
        "loss": total_loss / total_samples if total_samples else 0.0,
        "forward_seconds": forward_seconds,
        "images_per_second": total_samples / forward_seconds if forward_seconds > 0 else 0.0,
    }


def _count_parameters(model, *, trainable_only: bool = False) -> int:
    """Count model parameters."""
    params = model.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return int(sum(param.numel() for param in params))


def _autocast_context(device: str, *, enabled: bool):
    """Return a torch autocast context without using deprecated CUDA-only APIs."""
    require_dependency("torch", torch)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        return torch.amp.autocast(device_type=device_type, enabled=bool(enabled))
    return torch.cuda.amp.autocast(enabled=bool(enabled))


def _ablation_markdown(report: dict[str, Any]) -> list[str]:
    """Render a compact ImageNet ablation report."""
    lines = [
        "# ImageNet-1K SonoGloRe-ConvNeXt V1 Ablation",
        "",
        f"- Root: `{report.get('root')}`",
        f"- Split: `{report.get('split')}`",
        f"- Dataset size: `{report.get('dataset_size')}`",
        f"- Max batches: `{report.get('max_batches')}`",
        "",
        "| Model | Top-1 | Top-5 | Images/s | Params |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("rows", []):
        metrics = row.get("metrics", {})
        if row.get("status") == "requires_training" or metrics is None:
            lines.append(
                f"| {row.get('name', row.get('id'))} | requires training | requires training | - | - |"
            )
        else:
            lines.append(
                f"| {row.get('name', row.get('id'))} | "
                f"{float(metrics.get('top1', 0.0)):.4f} | "
                f"{float(metrics.get('top5', 0.0)):.4f} | "
                f"{float(metrics.get('images_per_second', 0.0)):.2f} | "
                f"{int(row.get('parameter_count', 0))} |"
            )
    return lines
