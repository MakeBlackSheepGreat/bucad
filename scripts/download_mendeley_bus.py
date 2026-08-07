"""Download a public Mendeley breast-ultrasound dataset and build a manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATASETS = {
    "bus_uclm": "7fvgj4jsp7",
    "busi_whu": "k6cpmwybk3",
}


def _download_one(task: tuple[str, Path]) -> tuple[str, int]:
    url, destination = task
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return str(destination), destination.stat().st_size
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))
    last_error = None
    for _ in range(3):
        try:
            response = session.get(url, timeout=120)
            response.raise_for_status()
            destination.write_bytes(response.content)
            return str(destination), len(response.content)
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(f"Download failed for {url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Mendeley breast ultrasound dataset.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output-root", default="data/external")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    dataset_id = args.dataset
    mendeley_id = DATASETS[dataset_id]
    output = Path(args.output_root).resolve() / dataset_id
    folders = requests.get(
        f"https://data.mendeley.com/public-api/datasets/{mendeley_id}/folders/1?version=1",
        timeout=60,
    ).json()
    tasks: list[tuple[str, Path]] = []
    rows: list[dict[str, str]] = []
    for folder in folders:
        files = requests.get(
            f"https://data.mendeley.com/public-api/datasets/{mendeley_id}/files?folder_id={folder['id']}&version=1",
            headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
            timeout=60,
        ).json()
        folder_output = output / folder["name"]
        for item in files:
            destination = folder_output / item["filename"]
            tasks.append((item["content_details"]["download_url"], destination))
            rows.append({"folder": folder["name"], "filename": item["filename"], "path": str(destination)})
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for future in concurrent.futures.as_completed([executor.submit(_download_one, task) for task in tasks]):
            try:
                future.result()
            except Exception as exc:
                failures.append(str(exc))
    manifest_path = output / "mendeley_files.csv"
    output.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["folder", "filename", "path"])
        writer.writeheader()
        writer.writerows(rows)
    if failures:
        (output / "download_failures.txt").write_text("\n".join(failures), encoding="utf-8")
    print(f"downloaded={len(tasks) - len(failures)} failed={len(failures)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
