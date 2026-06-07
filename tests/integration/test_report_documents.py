"""Integration tests for report documents."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from docx import Document

from src.utils.document_reports import export_report_documents


TEST_ROOT = Path("artifacts/test-workspace/test_report_documents")


def test_report_document_export_writes_readable_docx() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    reports_dir = TEST_ROOT / "reports"
    output_dir = reports_dir / "documents"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "model_pipeline_progress.md").write_text(
        "\n".join(
            [
                "# 模型流程进度同步",
                "",
                "- 当前已完成可演示原型和报告文档导出。",
            ]
        ),
        encoding="utf-8",
    )
    (reports_dir / "final_validation.md").write_text(
        "\n".join(
            [
                "# 最终验证报告",
                "",
                "- 诊断流程可以输出良恶性概率。",
                "",
                "| 项目 | 结果 |",
                "| --- | --- |",
                "| AUC | 0.8123 |",
            ]
        ),
        encoding="utf-8",
    )
    (reports_dir / "busi_eval.json").write_text(
        json.dumps(
            {
                "sample_count": 4,
                "metrics": {
                    "auc": 0.8123,
                    "threshold": 0.5,
                    "sensitivity": 0.75,
                    "specificity": 0.8,
                    "accuracy": 0.78,
                    "confusion": {"tn": 2, "fp": 0, "fn": 1, "tp": 1},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    outputs = export_report_documents(reports_dir, output_dir, basename="sample_report")

    docx_path = outputs["docx"]
    assert sorted(outputs) == ["docx"]
    assert docx_path.exists()
    assert docx_path.stat().st_size > 5_000

    document_text = "\n".join(paragraph.text for paragraph in Document(docx_path).paragraphs)
    assert "BUCAD 项目报告汇总" in document_text
    assert "模型流程进度同步" not in document_text
    assert "最终验证报告" in document_text
    assert "样本数量：4" in document_text


def test_report_document_export_rejects_pdf_format() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    reports_dir = TEST_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="Unsupported report format"):
        export_report_documents(reports_dir, reports_dir / "documents", formats=("pdf",))
