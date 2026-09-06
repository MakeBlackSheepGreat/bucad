"""Evaluate evidence-deletion behavior on frozen LesioNeXt-LEA OOF folds.

For each held-out BUSBRA image, the frozen matching-fold checkpoint first
produces an evidence map. The script then replaces the highest-evidence,
lowest-evidence, or a deterministic random set of equally sized spatial cells
with that image's per-channel mean. The primary endpoint is the decrease in the
true-class logit relative to the unmasked image. This is a post hoc mechanism
analysis only and does not affect model, checkpoint, threshold, or data choice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.busbra import BUSBRAClassificationLENSDataSet, load_busbra_manifest
from src.models.classifier import load_classifier
from src.preprocess.transforms import build_classifier_transform
from src.utils.config import load_project_config
from src.utils.runtime import optional_import


torch = optional_import("torch")
F = optional_import("torch.nn.functional")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "classifier" / "lesionext_lens_v1a_evidence_only.yml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "paper_evidence",
    )
    parser.add_argument(
        "--top-cell-count",
        type=int,
        default=10,
        help="Number of final evidence-map cells removed per image for every condition.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_812)
    parser.add_argument(
        "--oof-tolerance",
        type=float,
        default=1e-4,
        help="Maximum permitted absolute difference from the frozen OOF malignant score.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def stable_random_indices(sample_id: str, cell_count: int, selected_count: int) -> np.ndarray:
    payload = hashlib.sha256(sample_id.encode("utf-8")).digest()
    seed = int.from_bytes(payload[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed).choice(cell_count, size=selected_count, replace=False)


def spatial_masks(evidence_logits, sample_ids: list[str], image_height: int, image_width: int, selected_count: int):
    evidence_weights = torch.softmax(evidence_logits.flatten(1), dim=1)
    batch_size, cell_count = evidence_weights.shape
    if not 1 <= selected_count < cell_count:
        raise ValueError(f"--top-cell-count must fall in [1, {cell_count - 1}], received {selected_count}.")
    high_indices = torch.topk(evidence_weights, k=selected_count, dim=1, largest=True).indices
    low_indices = torch.topk(evidence_weights, k=selected_count, dim=1, largest=False).indices
    masks: dict[str, object] = {}
    for name, indices in (("high", high_indices), ("low", low_indices)):
        flat_mask = torch.zeros_like(evidence_weights, dtype=torch.bool)
        flat_mask.scatter_(1, indices, True)
        mask = flat_mask.reshape_as(evidence_logits)
        masks[name] = F.interpolate(mask.float(), size=(image_height, image_width), mode="nearest").bool()
    random_mask = torch.zeros_like(evidence_weights, dtype=torch.bool)
    for index, sample_id in enumerate(sample_ids):
        random_indices = torch.as_tensor(
            stable_random_indices(str(sample_id), cell_count, selected_count),
            device=evidence_logits.device,
            dtype=torch.long,
        )
        random_mask[index, random_indices] = True
    masks["random"] = F.interpolate(
        random_mask.reshape_as(evidence_logits).float(),
        size=(image_height, image_width),
        mode="nearest",
    ).bool()
    return masks


def mask_with_image_mean(images, mask):
    replacement = images.mean(dim=(2, 3), keepdim=True)
    return torch.where(mask.expand_as(images), replacement, images)


def cluster_bootstrap_mean_difference(
    rows: list[dict[str, object]],
    left_key: str,
    right_key: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    case_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        case_to_indices.setdefault(str(row["case_id"]), []).append(index)
    groups = [np.asarray(case_to_indices[case_id], dtype=np.int64) for case_id in sorted(case_to_indices)]
    left = np.asarray([float(row[left_key]) for row in rows], dtype=np.float64)
    right = np.asarray([float(row[right_key]) for row in rows], dtype=np.float64)
    observed = float(np.mean(left - right))
    random = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = random.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[group] for group in selected])
        differences[replicate] = float(np.mean(left[indices] - right[indices]))
    return {
        "observed_mean_difference": observed,
        "ci95_lower": float(np.quantile(differences, 0.025)),
        "ci95_upper": float(np.quantile(differences, 0.975)),
        "case_count": len(groups),
        "image_count": len(rows),
        "bootstrap_replicates": replicates,
    }


def main() -> int:
    args = parse_args()
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for evidence-deletion analysis.")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive.")
    config, paths = load_project_config(args.config)
    data_cfg = config.get("data", {})
    preprocess_cfg = data_cfg.get("preprocess", {}) or {}
    transform = build_classifier_transform(
        image_size=int(data_cfg.get("image_size", 224)),
        apply_clahe_enabled=bool(preprocess_cfg.get("clahe", False)),
        mean=preprocess_cfg.get("mean"),
        std=preprocess_cfg.get("std"),
        interpolation=str(preprocess_cfg.get("interpolation", "area")),
        crop_pct=float(preprocess_cfg.get("crop_pct", 1.0)),
    )
    manifest = load_busbra_manifest(paths.busbra_root)
    split_path = resolve_path(config["training"]["split_path"], paths.project_root)
    split = pd.read_csv(split_path)
    oof_path = PROJECT_ROOT / "artifacts" / "reports" / "lesionext_lens_v1a_evidence_only_5fold_oof.json"
    oof_rows = json.loads(oof_path.read_text(encoding="utf-8")).get("rows", [])
    expected_by_sample = {str(row["sample_id"]): row for row in oof_rows}
    if len(expected_by_sample) != len(oof_rows):
        raise ValueError("Frozen OOF predictions contain duplicated sample identifiers.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    observed_sample_ids: set[str] = set()
    maximum_oof_absolute_difference = 0.0
    oof_absolute_differences: list[float] = []
    frozen_probabilities: list[float] = []
    reproduced_probabilities: list[float] = []
    reproduction_labels: list[int] = []
    for fold_id in range(1, 6):
        validation_ids = set(
            split.loc[(split["fold_id"] == fold_id) & (split["stage"] == "val"), "sample_id"].astype(str)
        )
        validation_manifest = manifest[manifest["sample_id"].astype(str).isin(validation_ids)].copy()
        dataset = BUSBRAClassificationLENSDataSet(
            validation_manifest,
            image_size=int(data_cfg.get("image_size", 224)),
            transform=transform,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(data_cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=0,
        )
        checkpoint = PROJECT_ROOT / "artifacts" / "checkpoints" / f"lesionext_lens_v1a_evidence_only_fold{fold_id}.pt"
        model = load_classifier(dict(config["model"]), checkpoint_path=checkpoint, map_location="cpu").to(device).eval()
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device=device, dtype=torch.float32)
                true_labels = batch["label"].to(device=device, dtype=torch.long)
                sample_ids = [str(item) for item in batch["sample_id"]]
                original_logits = model(images)
                original_probabilities = torch.softmax(original_logits, dim=1)[:, 1].detach().cpu().numpy()
                evidence_logits = getattr(model, "last_evidence_map", None)
                if evidence_logits is None:
                    raise RuntimeError("The frozen LEA checkpoint did not expose an evidence map.")
                masks = spatial_masks(
                    evidence_logits,
                    sample_ids,
                    images.shape[-2],
                    images.shape[-1],
                    int(args.top_cell_count),
                )
                original_true_logits = original_logits.gather(1, true_labels[:, None]).squeeze(1)
                deleted_true_logits = {}
                for name, mask in masks.items():
                    masked_logits = model(mask_with_image_mean(images, mask))
                    deleted_true_logits[name] = masked_logits.gather(1, true_labels[:, None]).squeeze(1)
                evidence_area_ratio = float(args.top_cell_count / evidence_logits.flatten(1).shape[1])
                for index, sample_id in enumerate(sample_ids):
                    expected = expected_by_sample.get(sample_id)
                    if expected is None:
                        raise ValueError(f"Sample {sample_id} is absent from the frozen OOF predictions.")
                    expected_label = int(str(expected["pathology_label"]) == "malignant")
                    if int(expected["fold_id"]) != fold_id or expected_label != int(true_labels[index].item()):
                        raise ValueError(f"Frozen OOF identity mismatch for sample {sample_id}.")
                    absolute_difference = abs(float(original_probabilities[index]) - float(expected["malignant_probability"]))
                    maximum_oof_absolute_difference = max(maximum_oof_absolute_difference, absolute_difference)
                    oof_absolute_differences.append(absolute_difference)
                    frozen_probabilities.append(float(expected["malignant_probability"]))
                    reproduced_probabilities.append(float(original_probabilities[index]))
                    reproduction_labels.append(expected_label)
                    observed_sample_ids.add(sample_id)
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "case_id": str(batch["case_id"][index]),
                            "fold_id": fold_id,
                            "label": int(true_labels[index].item()),
                            "evidence_cell_area_ratio": evidence_area_ratio,
                            "high_evidence_true_logit_drop": float(original_true_logits[index] - deleted_true_logits["high"][index]),
                            "low_evidence_true_logit_drop": float(original_true_logits[index] - deleted_true_logits["low"][index]),
                            "random_evidence_true_logit_drop": float(original_true_logits[index] - deleted_true_logits["random"][index]),
                        }
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (int(row["fold_id"]), str(row["sample_id"])))
    if observed_sample_ids != set(expected_by_sample):
        raise ValueError("The evidence-deletion sample set does not exactly match the frozen OOF predictions.")
    if maximum_oof_absolute_difference > args.oof_tolerance:
        raise ValueError(
            "Frozen OOF reproduction failed. "
            f"Maximum absolute malignant-score difference was {maximum_oof_absolute_difference:.3e}, "
            f"which exceeds tolerance {args.oof_tolerance:.3e}."
        )
    frozen_probability_array = np.asarray(frozen_probabilities, dtype=np.float64)
    reproduced_probability_array = np.asarray(reproduced_probabilities, dtype=np.float64)
    reproduction_label_array = np.asarray(reproduction_labels, dtype=np.int8)
    threshold_mismatch_count = int(
        np.sum((frozen_probability_array >= 0.5) != (reproduced_probability_array >= 0.5))
    )
    summary = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (
            "high_evidence_true_logit_drop",
            "low_evidence_true_logit_drop",
            "random_evidence_true_logit_drop",
        )
    }
    comparisons = {
        "high_minus_low": cluster_bootstrap_mean_difference(
            rows,
            "high_evidence_true_logit_drop",
            "low_evidence_true_logit_drop",
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
        "high_minus_random": cluster_bootstrap_mean_difference(
            rows,
            "high_evidence_true_logit_drop",
            "random_evidence_true_logit_drop",
            replicates=args.bootstrap_replicates,
            seed=args.seed + 1,
        ),
    }
    report = {
        "scope": "Post hoc frozen BUSBRA OOF evidence-deletion analysis. No external cohort was read.",
        "analysis_unit": "image-level true-class logit changes with case-cluster bootstrap uncertainty",
        "device": str(device),
        "fold_count": 5,
        "image_count": len(rows),
        "case_count": len({str(row["case_id"]) for row in rows}),
        "oof_reproduction": {
            "source": str(oof_path.relative_to(PROJECT_ROOT)),
            "mean_absolute_malignant_score_difference": float(np.mean(oof_absolute_differences)),
            "p95_absolute_malignant_score_difference": float(np.percentile(oof_absolute_differences, 95)),
            "maximum_absolute_malignant_score_difference": maximum_oof_absolute_difference,
            "frozen_auc": float(roc_auc_score(reproduction_label_array, frozen_probability_array)),
            "reproduced_auc": float(roc_auc_score(reproduction_label_array, reproduced_probability_array)),
            "threshold_050_mismatch_count": threshold_mismatch_count,
            "tolerance": args.oof_tolerance,
            "status": "passed",
        },
        "masking_protocol": {
            "evidence_source": "spatial softmax of the original unmasked forward-pass evidence logits",
            "selected_cells_per_image": int(args.top_cell_count),
            "evidence_map_area_ratio": float(rows[0]["evidence_cell_area_ratio"]),
            "conditions": {
                "high": "highest-evidence cells",
                "low": "lowest-evidence cells",
                "random": "deterministic sample-id-seeded random cells",
            },
            "replacement": "per-image per-channel spatial mean after the frozen preprocessing pipeline",
            "endpoint": "original true-class logit minus masked true-class logit",
        },
        "mean_true_logit_drop": summary,
        "case_cluster_bootstrap_mean_differences": comparisons,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lesionext_lea_evidence_deletion_rows.csv"
    json_path = output_dir / "lesionext_lea_evidence_deletion.json"
    md_path = output_dir / "lesionext_lea_evidence_deletion.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# LesioNeXt-LEA Evidence-Deletion Analysis",
                "",
                "- Scope: post hoc analysis of frozen BUSBRA OOF folds. External cohorts were not read.",
                f"- Unit: `{report['image_count']}` images from `{report['case_count']}` cases. Every condition removed `{args.top_cell_count}` of 49 evidence cells per image.",
                f"- Mean true-class logit drop after high-evidence deletion: `{summary['high_evidence_true_logit_drop']:.4f}`.",
                f"- Mean true-class logit drop after low-evidence deletion: `{summary['low_evidence_true_logit_drop']:.4f}`.",
                f"- Mean true-class logit drop after random deletion: `{summary['random_evidence_true_logit_drop']:.4f}`.",
                f"- High minus low case-cluster mean difference: `{comparisons['high_minus_low']['observed_mean_difference']:+.4f}` [{comparisons['high_minus_low']['ci95_lower']:+.4f}, {comparisons['high_minus_low']['ci95_upper']:+.4f}].",
                f"- High minus random case-cluster mean difference: `{comparisons['high_minus_random']['observed_mean_difference']:+.4f}` [{comparisons['high_minus_random']['ci95_lower']:+.4f}, {comparisons['high_minus_random']['ci95_upper']:+.4f}].",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
