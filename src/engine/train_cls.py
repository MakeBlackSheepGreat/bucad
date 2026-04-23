from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.busbra import BUSBRAClassificationDataset, generate_busbra_split_assignments, load_busbra_manifest
from src.models.classifier import create_classifier
from src.utils.config import load_project_config
from src.utils.logging import get_logger
from src.utils.metrics import classification_metrics
from src.utils.reporting import write_json_report
from src.utils.runtime import ensure_dir, optional_import, require_dependency, seed_everything, select_device


torch = optional_import("torch")
optim = optional_import("torch.optim")
torch_utils_data = optional_import("torch.utils.data")


def _load_or_create_splits(
    manifest: pd.DataFrame, split_path: Path, *, fold_count: int, seed: int
) -> pd.DataFrame:
    if split_path.exists():
        return pd.read_csv(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    assignments = generate_busbra_split_assignments(
        manifest, n_splits=fold_count, seed=seed
    )
    assignments.to_csv(split_path, index=False)
    return assignments


def _split_manifest_for_fold(
    manifest: pd.DataFrame, assignments: pd.DataFrame, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_assignments = assignments[assignments["fold_id"] == fold]
    train_ids = set(fold_assignments.loc[fold_assignments["stage"] == "train", "sample_id"])
    val_ids = set(fold_assignments.loc[fold_assignments["stage"] == "val", "sample_id"])
    train_manifest = manifest[manifest["sample_id"].isin(train_ids)].reset_index(drop=True)
    val_manifest = manifest[manifest["sample_id"].isin(val_ids)].reset_index(drop=True)
    return train_manifest, val_manifest


def _evaluate_model(model, loader, device: str) -> dict[str, Any]:
    require_dependency("torch", torch)
    model.eval()
    y_true: list[int] = []
    malignant_probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            malignant_probabilities.extend(probs[:, 1].cpu().tolist())
    return classification_metrics(y_true, malignant_probabilities)


def run_classifier_training(
    config_path: str | Path,
    *,
    fold: int = 1,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    require_dependency("torch", torch)
    require_dependency("torch.optim", optim)
    require_dependency("torch.utils.data", torch_utils_data)

    config, paths = load_project_config(config_path)
    logger = get_logger("train_cls")
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = select_device(str(config.get("device", "auto")))

    manifest = load_busbra_manifest(paths.busbra_root)
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    split_path = Path(training_cfg.get("split_path", paths.reports_root / "busbra_5fold_splits.csv"))
    if not split_path.is_absolute():
        split_path = (paths.project_root / split_path).resolve()
    assignments = _load_or_create_splits(
        manifest,
        split_path,
        fold_count=int(training_cfg.get("fold_count", 5)),
        seed=seed,
    )
    train_manifest, val_manifest = _split_manifest_for_fold(manifest, assignments, fold)
    if train_manifest.empty or val_manifest.empty:
        raise ValueError(f"Fold {fold} produced an empty train or validation split.")

    train_dataset = BUSBRAClassificationDataset(
        train_manifest, image_size=int(data_cfg.get("image_size", 224))
    )
    val_dataset = BUSBRAClassificationDataset(
        val_manifest, image_size=int(data_cfg.get("image_size", 224))
    )
    train_loader = torch_utils_data.DataLoader(
        train_dataset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    val_loader = torch_utils_data.DataLoader(
        val_dataset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    model = create_classifier(
        model_name=config.get("model", {}).get("name", "resnet18"),
        pretrained=bool(config.get("model", {}).get("pretrained", True)),
        in_chans=int(config.get("model", {}).get("in_chans", 3)),
        num_classes=int(config.get("model", {}).get("num_classes", 2)),
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    criterion = torch.nn.CrossEntropyLoss()

    epochs = int(epochs_override or training_cfg.get("epochs", 5))
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        logger.info("fold=%s epoch=%s loss=%.4f", fold, epoch + 1, np.mean(losses))

    metrics = _evaluate_model(model, val_loader, device)
    checkpoints_dir = ensure_dir(paths.checkpoints_root)
    reports_dir = ensure_dir(paths.reports_root)
    checkpoint_path = checkpoints_dir / output_cfg.get("checkpoint_name", "classifier_fold{fold}.pt").format(fold=fold)
    report_path = reports_dir / output_cfg.get("report_name", "train_cls_fold{fold}.json").format(fold=fold)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": config.get("model", {}),
            "fold": fold,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    report = {
        "fold": fold,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "train_size": int(len(train_manifest)),
        "val_size": int(len(val_manifest)),
    }
    write_json_report(report_path, report)
    return report
