"""Standardise the Kaggle MIMIC-CXR mirror into a flat schema this repo can
consume:

  data/mimic-cxr/
    images/             flat JPEGs named "{study_id}__{dicom_id}.jpg"
    reports.csv         columns: study_id,subject_id,text,image_path

Resizes images so the short side equals `dataset.image_short_side`
(default 512) — both ColPali and MedGemma down-sample further, but this
keeps disk usage manageable while preserving enough detail.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .common import Paths, ensure_dir, get_logger, load_config

log = get_logger("preprocess")


def discover_kaggle_layout(data_root: Path) -> tuple[Path, pd.DataFrame]:
    """Find the images dir and the reports CSV inside the unzipped Kaggle dump.

    The Kaggle mirror's layout has shifted across uploads; this function tries
    a few known shapes and falls back to a heuristic search.
    """
    candidate_csvs = list(data_root.rglob("*.csv"))
    reports_csv: Path | None = None
    for c in candidate_csvs:
        # Heuristic: report tables contain a `text` column or `findings` column.
        try:
            head = pd.read_csv(c, nrows=2)
        except Exception:
            continue
        cols = {c.lower() for c in head.columns}
        if "text" in cols or ("findings" in cols and "subject_id" in cols):
            reports_csv = c
            break
    if reports_csv is None:
        raise FileNotFoundError(
            f"No reports CSV with a `text` column found under {data_root}. "
            "Inspect the unzipped Kaggle data manually."
        )
    df = pd.read_csv(reports_csv)
    df.columns = [c.lower() for c in df.columns]

    images_root: Path | None = None
    for p in data_root.rglob("*"):
        if p.is_dir() and any(c.suffix.lower() in {".jpg", ".jpeg", ".png"}
                              for c in p.iterdir() if c.is_file()):
            images_root = p
            break
    if images_root is None:
        # Some mirrors store everything in one flat dir; the parent of any
        # *.jpg works.
        for p in data_root.rglob("*.jpg"):
            images_root = p.parent
            break
    if images_root is None:
        raise FileNotFoundError(f"No image files found under {data_root}")

    log.info("Discovered reports CSV: %s", reports_csv)
    log.info("Discovered images root: %s", images_root)
    return images_root, df


def _ensure_text_column(df: pd.DataFrame) -> pd.DataFrame:
    if "text" in df.columns:
        return df
    # Synthesise `text` from findings + impression if needed.
    fields = [c for c in ("findings", "impression") if c in df.columns]
    if not fields:
        raise ValueError("Reports CSV must contain `text` or findings/impression.")
    df = df.copy()
    df["text"] = (
        df[fields].fillna("").astype(str).agg(
            lambda r: "\n\n".join(f"{k.upper()}: {v}" for k, v in r.items() if v.strip()),
            axis=1,
        )
    )
    return df


def _resize_one(args) -> tuple[str, str] | None:
    src, dst, short_side = args
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = short_side
            if min(w, h) > s:
                if w < h:
                    new = (s, int(h * s / w))
                else:
                    new = (int(w * s / h), s)
                im = im.resize(new, Image.LANCZOS)
            im.save(dst, "JPEG", quality=92)
        return str(src), str(dst)
    except Exception as e:
        log.warning("Skipping %s: %s", src, e)
        return None


def preprocess(
    images_root: Path,
    df_reports: pd.DataFrame,
    out_images_dir: Path,
    out_reports_csv: Path,
    short_side: int,
    num_workers: int = 8,
) -> None:
    ensure_dir(out_images_dir)
    df_reports = _ensure_text_column(df_reports)

    # Build a mapping from study_id → image path(s). The Kaggle mirror
    # typically uses `s{study_id}` or `study_{study_id}` filenames; we
    # match by substring.
    image_paths = list(images_root.rglob("*.jpg")) + list(images_root.rglob("*.png"))
    by_basename = {p.stem: p for p in image_paths}

    rows_out: list[dict] = []
    resize_jobs: list[tuple[Path, Path, int]] = []
    for _, row in df_reports.iterrows():
        study_id = str(row.get("study_id") or row.get("study", "")).strip()
        subject_id = str(row.get("subject_id", "")).strip()
        text = str(row.get("text") or "").strip()
        if not study_id or not text:
            continue

        # Find any image whose name contains the study_id.
        matches = [p for k, p in by_basename.items() if study_id in k]
        if not matches:
            continue
        for src in matches:
            dst = out_images_dir / f"{study_id}__{src.stem}.jpg"
            if not dst.exists():
                resize_jobs.append((src, dst, short_side))
            rows_out.append({
                "study_id":   study_id,
                "subject_id": subject_id,
                "image_path": str(dst),
                "text":       text,
            })

    log.info("Resizing %d images with %d workers", len(resize_jobs), num_workers)
    if resize_jobs:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = [ex.submit(_resize_one, j) for j in resize_jobs]
            for _ in tqdm(as_completed(futs), total=len(futs)):
                pass

    out_reports_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_reports_csv, index=False)
    log.info("Wrote %d rows to %s", len(rows_out), out_reports_csv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    images_root, df = discover_kaggle_layout(paths.data_root)
    preprocess(
        images_root      = images_root,
        df_reports       = df,
        out_images_dir   = paths.images_dir,
        out_reports_csv  = paths.reports_csv,
        short_side       = cfg["dataset"]["image_short_side"],
        num_workers      = args.workers,
    )


if __name__ == "__main__":
    main()
