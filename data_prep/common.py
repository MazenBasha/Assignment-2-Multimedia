"""Shared helpers for data_prep scripts: config loading, logging, IO."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "%H:%M:%S",
        ))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | os.PathLike, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@dataclass(frozen=True)
class Paths:
    data_root: Path
    images_dir: Path
    reports_csv: Path
    splits_dir: Path
    artifacts_dir: Path
    vqa_dataset: Path
    outputs_dir: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Paths":
        p = cfg["paths"]
        return cls(
            data_root     = Path(p["data_root"]),
            images_dir    = Path(p["images_dir"]),
            reports_csv   = Path(p["reports_csv"]),
            splits_dir    = Path(p["splits_dir"]),
            artifacts_dir = Path(p["artifacts_dir"]),
            vqa_dataset   = Path(p["vqa_dataset"]),
            outputs_dir   = Path(p["outputs_dir"]),
        )
