"""Unit tests for the SonoGloReNet ablation workflow."""

from __future__ import annotations

from pathlib import Path

from src.experiments import sonoglore_ablation
from src.experiments.sonoglore_ablation import (
    build_chinese_report,
    build_parser,
    dry_run,
    rank_candidates,
)


def test_rank_candidates_prefers_balanced_accuracy_then_auc_then_f1_then_internal_auc() -> None:
    """Verify the candidate sorter follows the fixed ranking rule."""
    rows = [
        {
            "candidate_id": "a",
            "candidate_name": "a",
            "train_metrics": {"auc": 0.80},
            "busi_metrics": {"balanced_accuracy": 0.81, "auc": 0.82, "f1_score": 0.70},
        },
        {
            "candidate_id": "b",
            "candidate_name": "b",
            "train_metrics": {"auc": 0.90},
            "busi_metrics": {"balanced_accuracy": 0.81, "auc": 0.82, "f1_score": 0.75},
        },
        {
            "candidate_id": "c",
            "candidate_name": "c",
            "train_metrics": {"auc": 0.95},
            "busi_metrics": {"balanced_accuracy": 0.84, "auc": 0.70, "f1_score": 0.60},
        },
    ]

    ranked = rank_candidates(rows)

    assert [row["candidate_id"] for row in ranked] == ["c", "b", "a"]


def test_sonoglore_ablation_dry_run_writes_report_skeletons(monkeypatch) -> None:
    """Verify dry-run validates config assembly and routes report writes."""
    writes: list[Path] = []

    def fake_write_json_report(path, payload):
        source = Path(path)
        destination = Path("C:/virtual") / source.parent.name / source.name
        writes.append(destination)
        return destination

    def fake_write_markdown_report(path, lines):
        source = Path(path)
        destination = Path("C:/virtual") / source.parent.name / source.name
        writes.append(destination)
        return destination

    monkeypatch.setattr(sonoglore_ablation, "write_json_report", fake_write_json_report)
    monkeypatch.setattr(sonoglore_ablation, "write_markdown_report", fake_write_markdown_report)

    args = build_parser().parse_args(["--dry-run"])

    report = dry_run(args)

    assert report["dry_run"] is True
    assert len(report["stage1_structure_rows"]) == 3
    assert report["stage1_structure_rows"][0]["train_source"] == "dry_run"
    assert report["report_paths"]["chinese_fold1"].endswith("sonoglore_ablation_matrix_fold1.md")
    assert report["report_paths"]["english_fold1"].endswith("sonoglore_ablation_matrix_fold1.md")
    assert report["report_paths"]["chinese_fold1"] != report["report_paths"]["english_fold1"]
    assert "chinese" in report["report_paths"]["chinese_fold1"].lower()
    assert "english" in report["report_paths"]["english_fold1"].lower()
    assert report["summary_json"].endswith("sonoglore_ablation_matrix_summary.json")
    assert len(writes) == 3
    assert all(path.name for path in writes)


def test_build_chinese_report_mentions_fold1_stop_when_gate_fails() -> None:
    """Verify the Chinese report renders the non-promotion wording."""
    lines = build_chinese_report(
        {
            "date": "2026-06-18",
            "references": [],
            "stage1_structure_ranking": [],
            "stage2_regularization_ranking": [],
            "promotion_gate": {
                "balanced_accuracy_threshold": 0.80,
                "minimum_auc": 0.87,
                "candidate_id": "S1_T0",
                "candidate_name": "demo",
                "balanced_accuracy": 0.79,
                "auc": 0.86,
                "passed": False,
            },
            "final_candidate": {
                "candidate_id": "S1_T0",
                "candidate_name": "demo",
                "config_path": "demo.yml",
                "inference_config": "demo_infer.yml",
                "train_metrics": {"auc": 0.9},
                "busi_metrics": {"auc": 0.86, "balanced_accuracy": 0.79},
            },
            "error_analysis": {},
            "latency": {"sample_id": "benign (1)", "warmup_runs": 2, "measured_runs": 10, "rows": []},
        }
    )

    text = "\n".join(lines)

    assert "终止在 fold1" in text
    assert "未进入 5-fold" in text
