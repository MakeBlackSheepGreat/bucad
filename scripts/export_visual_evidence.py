"""CLI for exporting segmentation and Grad-CAM visual evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.busi import load_busi_manifest
from src.engine.inference import BreastUltrasoundInferenceService
from src.preprocess.io import save_image
from src.utils.config import load_project_config
from src.utils.reporting import write_markdown_report


def export_visual_evidence(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int = 6,
    classifier_predictor: Callable[[np.ndarray], tuple[float, float]] | None = None,
    segmenter_predictor: Callable[[np.ndarray], np.ndarray] | None = None,
    explanation_generator: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, object]:
    config, paths = load_project_config(config_path)
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = (paths.project_root / destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    manifest = load_busi_manifest(paths.busi_root, include_normal=False).head(limit)
    service = BreastUltrasoundInferenceService(
        config.get("runtime", {}),
        paths=paths,
        classifier_predictor=classifier_predictor,
        segmenter_predictor=segmenter_predictor,
        explanation_generator=explanation_generator,
    )

    rows: list[dict[str, object]] = []
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        sample_dir = destination / f"{index:03d}_{row.sample_id.replace(' ', '_')}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        response = service.diagnose(row.image_path, input_filename=Path(row.image_path).name)
        if response.original_image_view is not None:
            save_image(sample_dir / "original.png", response.original_image_view)
        if response.lesion_overlay_view is not None:
            save_image(sample_dir / "lesion_overlay.png", response.lesion_overlay_view)
        if response.explanation_view is not None:
            save_image(sample_dir / "explanation.png", response.explanation_view)
        rows.append(
            {
                "sample_id": row.sample_id,
                "status": response.status,
                "final_label": response.result.final_label if response.result else None,
                "lesion_visualization_missing_reason": response.lesion_visualization_missing_reason,
                "explanation_missing_reason": response.explanation_missing_reason,
            }
        )

    review_path = paths.reports_root / "visual_evidence_review.md"
    write_markdown_report(review_path, _visual_review_lines(rows, destination))
    return {"output_dir": str(destination), "sample_count": len(rows), "rows": rows}


def _visual_review_lines(rows: list[dict[str, object]], output_dir: Path) -> list[str]:
    lines = [
        "# Visual Evidence Review",
        "",
        f"Output directory: `{output_dir}`",
        "",
        "| Sample | Status | Label | Lesion Missing Reason | Explanation Missing Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['status']} | {row.get('final_label') or ''} | "
            f"{row.get('lesion_visualization_missing_reason') or ''} | "
            f"{row.get('explanation_missing_reason') or ''} |"
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export report-ready visual evidence samples.")
    parser.add_argument("--config", default="configs/inference/demo.yml")
    parser.add_argument("--output-dir", default="artifacts/reports/visual_evidence")
    parser.add_argument("--limit", type=int, default=6)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = export_visual_evidence(args.config, args.output_dir, limit=args.limit)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
