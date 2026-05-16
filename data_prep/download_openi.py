"""Download the Indiana University chest X-ray dataset (Open-i, NLM).

Public, no Kaggle credentials needed. Two archives from openi.nlm.nih.gov:
  * NLMCXR_png.tgz       (~600 MB) — frontal + lateral X-rays
  * NLMCXR_reports.tgz   (~3 MB)   — paired XML reports

We download both, extract, parse the XMLs into a flat `reports.csv`, and
match each report to its frontal image. Skip lateral views to keep the
retrieval corpus consistent.

Output structure (matches what `data_prep.split` expects):
    data/openi/
      reports.csv      columns: study_id, subject_id, image_path, text
      images/          flat dir of frontal X-rays renamed by study_id
      raw/             raw archives + extracted tree (kept for debugging)
"""

from __future__ import annotations

import argparse
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

from .common import Paths, ensure_dir, get_logger, load_config

log = get_logger("download_openi")

OPENI_BASE = "https://openi.nlm.nih.gov/imgs/collections"
PNG_ARCHIVE     = f"{OPENI_BASE}/NLMCXR_png.tgz"
REPORTS_ARCHIVE = f"{OPENI_BASE}/NLMCXR_reports.tgz"


def _stream_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        log.info("Already downloaded: %s (%d MB)", dest, dest.stat().st_size >> 20)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    with urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with dest.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                bar.update(len(buf))


def _extract(archive: Path, dest_dir: Path) -> None:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        log.info("Already extracted: %s", dest_dir)
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s -> %s", archive, dest_dir)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest_dir)


def _section_text(report_root: ET.Element, label: str) -> str:
    """Pull the text of <AbstractText Label="FINDINGS">...</AbstractText> nodes."""
    for node in report_root.iter("AbstractText"):
        if (node.get("Label") or "").upper() == label.upper():
            return (node.text or "").strip()
    return ""


def _parse_report_xml(path: Path) -> dict | None:
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    findings   = _section_text(root, "FINDINGS")
    impression = _section_text(root, "IMPRESSION")
    indication = _section_text(root, "INDICATION") or _section_text(root, "COMPARISON")
    if not (findings or impression):
        return None

    # Frontal image filenames are referenced like <parentImage id="CXR123_IM-0001-1001"/>.
    image_ids = [p.get("id") for p in root.iter("parentImage") if p.get("id")]
    if not image_ids:
        return None

    text_parts = []
    if indication:
        text_parts.append(f"INDICATION: {indication}")
    if findings:
        text_parts.append(f"FINDINGS: {findings}")
    if impression:
        text_parts.append(f"IMPRESSION: {impression}")
    text = "\n\n".join(text_parts)

    # uid: derive from filename ("1.xml" -> "1") and ALSO carry subject from filename pattern.
    study_id = path.stem.lstrip("CXR0") or path.stem
    return {
        "study_id":   study_id,
        "subject_id": study_id,             # Open-i has 1:1 patient/study, so reuse.
        "image_ids":  image_ids,
        "text":       text,
    }


def _find_image_file(image_id: str, images_root: Path) -> Path | None:
    # Open-i PNG files are named like CXR1_IM-0001-1001.png (sometimes .png omitted in XML).
    candidates = [
        images_root / f"{image_id}.png",
        images_root / image_id,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: brute-force search the tree.
    for p in images_root.rglob(f"{image_id}.png"):
        return p
    return None


def _copy_and_rename(src: Path, dst: Path) -> None:
    if not dst.exists():
        shutil.copy(src, dst)


def build_reports_csv(
    extracted_reports_dir: Path,
    extracted_images_dir: Path,
    out_images_dir: Path,
    out_reports_csv: Path,
    workers: int = 4,
) -> int:
    ensure_dir(out_images_dir)
    xml_files = list(extracted_reports_dir.rglob("*.xml"))
    log.info("Found %d XML reports", len(xml_files))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_parse_report_xml, p): p for p in xml_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="parse xml"):
            parsed = fut.result()
            if not parsed:
                continue
            for img_id in parsed["image_ids"]:
                src_img = _find_image_file(img_id, extracted_images_dir)
                if src_img is None:
                    continue
                dst_img = out_images_dir / f"{parsed['study_id']}__{img_id}.png"
                _copy_and_rename(src_img, dst_img)
                rows.append({
                    "study_id":   parsed["study_id"],
                    "subject_id": parsed["subject_id"],
                    "image_path": str(dst_img),
                    "text":       parsed["text"],
                })
                # one image per study is enough for our small corpus
                break

    out_reports_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_reports_csv, index=False)
    log.info("Wrote %d rows -> %s", len(rows), out_reports_csv)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--skip-download", action="store_true",
                    help="Use already-downloaded archives in data/openi/raw/.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    raw_dir = paths.data_root / "raw"
    extracted_reports = raw_dir / "reports_extracted"
    extracted_images  = raw_dir / "images_extracted"
    reports_tgz = raw_dir / "NLMCXR_reports.tgz"
    images_tgz  = raw_dir / "NLMCXR_png.tgz"

    if not args.skip_download:
        _stream_download(REPORTS_ARCHIVE, reports_tgz)
        _stream_download(PNG_ARCHIVE,     images_tgz)

    _extract(reports_tgz, extracted_reports)
    _extract(images_tgz,  extracted_images)

    n = build_reports_csv(
        extracted_reports_dir = extracted_reports,
        extracted_images_dir  = extracted_images,
        out_images_dir        = paths.images_dir,
        out_reports_csv       = paths.reports_csv,
    )
    log.info("Open-i ingestion done. %d (image, report) rows.", n)


if __name__ == "__main__":
    main()
