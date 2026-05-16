"""Patient-level train/test split.

Crucially, `subject_id` (the patient) is partitioned, NOT `study_id` —
the same patient must never appear in both train (the retrieval corpus)
and test (held-out eval). This prevents trivial near-duplicate retrieval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Paths, ensure_dir, get_logger, load_config

log = get_logger("split")


def split_by_patient(
    reports_csv: Path,
    splits_dir: Path,
    test_fraction: float,
    seed: int,
    max_train: int | None = None,
    max_test: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(reports_csv)
    df = df.dropna(subset=["text", "image_path"]).reset_index(drop=True)
    log.info("Loaded %d rows / %d unique patients",
             len(df), df["subject_id"].nunique())

    rng = np.random.default_rng(seed)
    patients = df["subject_id"].dropna().unique()
    rng.shuffle(patients)
    n_test_patients = max(1, int(round(len(patients) * test_fraction)))
    test_patients = set(patients[:n_test_patients].tolist())

    train_df = df[~df["subject_id"].isin(test_patients)].copy()
    test_df  = df[df["subject_id"].isin(test_patients)].copy()

    if max_train is not None and len(train_df) > max_train:
        train_df = train_df.sample(n=max_train, random_state=seed).reset_index(drop=True)
    if max_test is not None and len(test_df) > max_test:
        test_df = test_df.sample(n=max_test, random_state=seed).reset_index(drop=True)

    ensure_dir(splits_dir)
    train_path = splits_dir / "train.parquet"
    test_path  = splits_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,  index=False)
    log.info("Train: %d rows (%d patients) → %s",
             len(train_df), train_df["subject_id"].nunique(), train_path)
    log.info("Test : %d rows (%d patients) → %s",
             len(test_df),  test_df["subject_id"].nunique(),  test_path)

    overlap = set(train_df["subject_id"]) & set(test_df["subject_id"])
    if overlap:
        raise RuntimeError(f"Patient leakage detected: {len(overlap)} subjects in both splits.")
    return train_df, test_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    split_by_patient(
        reports_csv   = paths.reports_csv,
        splits_dir    = paths.splits_dir,
        test_fraction = cfg["dataset"]["test_fraction"],
        seed          = cfg["dataset"]["seed"],
        max_train     = cfg["dataset"].get("max_train_corpus"),
        max_test      = cfg["dataset"].get("max_test_examples"),
    )


if __name__ == "__main__":
    main()
