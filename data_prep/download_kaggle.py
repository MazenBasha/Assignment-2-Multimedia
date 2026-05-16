"""Download the MIMIC-CXR Kaggle mirror.

Dataset: https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset

Usage:
    python -m data_prep.download_kaggle --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .common import Paths, ensure_dir, get_logger, load_config

KAGGLE_SLUG = "simhadrisadaram/mimic-cxr-dataset"
log = get_logger("download_kaggle")


def kaggle_cli_available() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download(target_dir: Path) -> None:
    ensure_dir(target_dir)
    if not kaggle_cli_available():
        raise RuntimeError(
            "Kaggle CLI not installed or not on PATH. "
            "Run `pip install kaggle` and place kaggle.json in ~/.kaggle/."
        )
    log.info("Downloading %s -> %s", KAGGLE_SLUG, target_dir)
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_SLUG,
            "-p", str(target_dir),
            "--unzip",
        ],
        check=True,
    )
    log.info("Done. Inspect with `ls %s`.", target_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    download(paths.data_root)


if __name__ == "__main__":
    main()
