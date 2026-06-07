"""Utility script for convnext seed soup workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.runtime import optional_import


torch = optional_import("torch")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="Average matching ConvNeXt-Tiny seed checkpoints fold by fold."
    )
    parser.add_argument(
        "--base-template",
        default="artifacts/checkpoints/convnext_tiny_timm_recipe_fold{fold}.pt",
    )
    parser.add_argument(
        "--candidate-template",
        default="artifacts/checkpoints/convnext_tiny_timm_recipe_seed123_fold{fold}.pt",
    )
    parser.add_argument(
        "--output-template",
        default="artifacts/checkpoints/convnext_tiny_timm_recipe_soup42_123_a050_fold{fold}.pt",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--fold-count", type=int, default=5)
    return parser


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load checkpoint."""
    if torch is None:
        raise RuntimeError("Torch is required to build checkpoint soups.")
    checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a state_dict: {path}")
    return checkpoint


def _average_state_dicts(
    base_state: dict[str, Any],
    candidate_state: dict[str, Any],
    *,
    alpha: float,
) -> dict[str, Any]:
    """Average compatible checkpoint state dictionaries."""
    if set(base_state) != set(candidate_state):
        missing = sorted(set(base_state).symmetric_difference(candidate_state))
        raise ValueError(f"Checkpoint state_dict keys do not match: {missing[:5]}")
    alpha = float(alpha)
    output = {}
    for key in base_state:
        base_value = base_state[key]
        candidate_value = candidate_state[key]
        if not hasattr(base_value, "dtype") or not hasattr(candidate_value, "dtype"):
            output[key] = base_value
            continue
        if not base_value.dtype.is_floating_point:
            output[key] = base_value.clone()
            continue
        output[key] = (1.0 - alpha) * base_value + alpha * candidate_value
    return output


def run(args: argparse.Namespace) -> list[str]:
    """Run the experiment workflow and return the generated summary."""
    if torch is None:
        raise RuntimeError("Torch is required to build checkpoint soups.")
    alpha = float(args.alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    outputs: list[str] = []
    for fold in range(1, int(args.fold_count) + 1):
        base_path = Path(str(args.base_template).format(fold=fold))
        candidate_path = Path(str(args.candidate_template).format(fold=fold))
        output_path = Path(str(args.output_template).format(fold=fold))
        base_checkpoint = _load_checkpoint(base_path)
        candidate_checkpoint = _load_checkpoint(candidate_path)
        averaged = dict(base_checkpoint)
        averaged["state_dict"] = _average_state_dicts(
            base_checkpoint["state_dict"],
            candidate_checkpoint["state_dict"],
            alpha=alpha,
        )
        averaged["soup"] = {
            "method": "linear_weight_average",
            "base_checkpoint": str(base_path),
            "candidate_checkpoint": str(candidate_path),
            "alpha": alpha,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(averaged, output_path)
        outputs.append(str(output_path))
    return outputs


def main() -> int:
    """Parse CLI arguments and run the script entry point."""
    args = build_parser().parse_args()
    for path in run(args):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
