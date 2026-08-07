"""Resumable download of the BUSI-WHU Kaggle mirror for label inspection."""

from __future__ import annotations

from pathlib import Path

import requests


URL = "https://www.kaggle.com/api/v1/datasets/download/orvile/busi-whu-breast-cancer-ultrasound-image-dataset"


def main() -> int:
    destination = Path("data/external/busi_whu_kaggle.zip").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    response = requests.get(URL, headers=headers, stream=True, timeout=180)
    response.raise_for_status()
    mode = "ab" if existing and response.status_code == 206 else "wb"
    if mode == "wb":
        existing = 0
    with destination.open(mode) as handle:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                handle.write(chunk)
    print(f"downloaded={destination.stat().st_size} status={response.status_code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
