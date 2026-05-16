"""Split the constructed VQA dataset into train/test by patient.

The held-out VQA test set must only contain patients that are also in the
held-out report test set, so we never query images that ColPali has
indexed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import Paths, ensure_dir, get_logger, load_config

log = get_logger("split_vqa")


def split(vqa_parquet: Path, splits_dir: Path) -> None:
    df = pd.read_parquet(vqa_parquet)
    test_subjects = set(pd.read_parquet(splits_dir / "test.parquet")["subject_id"].astype(str))
    train_subjects = set(pd.read_parquet(splits_dir / "train.parquet")["subject_id"].astype(str))

    df_test  = df[df["subject_id"].astype(str).isin(test_subjects)].copy()
    df_train = df[df["subject_id"].astype(str).isin(train_subjects)].copy()

    ensure_dir(splits_dir)
    df_train.to_parquet(splits_dir / "vqa_train.parquet", index=False)
    df_test.to_parquet (splits_dir / "vqa_test.parquet",  index=False)
    log.info("VQA train: %d, test: %d", len(df_train), len(df_test))
    if not len(df_test):
        log.warning(
            "VQA test split is empty — probably your VQA generator only ran on "
            "training reports. That's expected; you may want to also generate "
            "VQA pairs over a small slice of the test split, or shrink the "
            "test fraction."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    split(paths.vqa_dataset, paths.splits_dir)


if __name__ == "__main__":
    main()
