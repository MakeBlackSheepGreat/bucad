from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.document_reports import export_report_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a readable Word DOCX report document.")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument("--output-dir", default="artifacts/reports/documents")
    parser.add_argument("--basename", default="bucad_report_summary")
    parser.add_argument("--formats", default="docx", help="Comma-separated values. Currently supported: docx")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    outputs = export_report_documents(
        args.reports_dir,
        args.output_dir,
        basename=args.basename,
        formats=formats,
    )
    for key, value in outputs.items():
        print(f"{key.upper()} report written: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
