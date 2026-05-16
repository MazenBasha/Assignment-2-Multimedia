"""Preprocess the `simhadrisadaram/mimic-cxr-dataset` Kaggle mirror.

This Kaggle mirror has a different schema from the canonical MIMIC-CXR-JPG:

  * One CSV row per **patient** (not per study).
  * `image`        — stringified Python list of all image paths for the patient.
  * `text`         — stringified Python list of reports, one per **study**, in
                     the same order studies first appear in `image`.
  * `PA`/`AP`/`Lateral` — stringified lists of view-specific image paths.
  * `text_augment` — paraphrased reports (we ignore these and keep ground truth).

Image paths in the CSV are RELATIVE to
`<data_root>/official_data_iccv_final/`, e.g.
`files/p10/p10000032/s50414267/02aa…jpg`.

Output schema (matches what `data_prep.split` expects):
    study_id, subject_id, image_path, text

We emit ONE row per study — picking the image with view preference
**PA > AP > Lateral > first available** — so retrieval/generation see a
consistent view per study and the corpus isn't bloated by 3× duplicates.

NOTE: we deliberately do NOT resize the images. The Kaggle dataset already
ships them as JPGs and `/kaggle/input` is read-only; ColPali and MedGemma
both downsample internally so a resize step would only waste disk.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .common import Paths, ensure_dir, get_logger, load_config

log = get_logger("preprocess_kaggle")


def _safe_literal_eval(x) -> list:
    """Parse a stringified Python list defensively; return [] on any failure."""
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except (SyntaxError, ValueError):
        return []


def _extract_study_id(img_rel_path: str) -> str | None:
    """Extract `sXXXXXXX` from a path like `files/p10/p10000032/s50414267/...jpg`."""
    for part in img_rel_path.split("/"):
        if part.startswith("s") and part[1:].isdigit():
            return part
    return None


def preprocess(
    data_root: Path,
    out_reports_csv: Path,
    include_validate: bool = True,
) -> int:
    files_root = data_root / "official_data_iccv_final"
    train_csv  = data_root / "mimic_cxr_aug_train.csv"
    val_csv    = data_root / "mimic_cxr_aug_validate.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Train CSV not found at {train_csv}. "
            "Check that the Kaggle dataset is attached and that data_root in "
            "configs/kaggle.yaml points at the right place."
        )

    csvs = [("train", train_csv)]
    if include_validate and val_csv.exists():
        csvs.append(("validate", val_csv))

    rows_out: list[dict] = []
    n_skipped_missing_text = 0
    n_skipped_missing_file = 0

    for split_name, csv_path in csvs:
        log.info("Loading %s split: %s", split_name, csv_path)
        df = pd.read_csv(csv_path)
        log.info("  %d patient rows", len(df))

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"parse {split_name}"):
            subject_id = str(row["subject_id"])
            images_all = _safe_literal_eval(row.get("image"))
            texts      = _safe_literal_eval(row.get("text"))
            pa_set     = set(_safe_literal_eval(row.get("PA")))
            ap_set     = set(_safe_literal_eval(row.get("AP")))
            lat_set    = set(_safe_literal_eval(row.get("Lateral")))

            # Group images by their study_id and remember first-seen order.
            by_study: dict[str, list[str]] = defaultdict(list)
            study_order: list[str] = []
            for img in images_all:
                sid = _extract_study_id(img)
                if not sid:
                    continue
                if sid not in by_study:
                    study_order.append(sid)
                by_study[sid].append(img)

            # `text[i]` corresponds to `study_order[i]`.
            for i, sid in enumerate(study_order):
                if i >= len(texts):
                    n_skipped_missing_text += 1
                    continue
                report = (texts[i] or "").strip()
                if not report:
                    n_skipped_missing_text += 1
                    continue

                # Pick one image per study with view preference: PA > AP > Lateral > first.
                imgs = by_study[sid]
                chosen = (
                    next((p for p in imgs if p in pa_set), None)
                    or next((p for p in imgs if p in ap_set), None)
                    or next((p for p in imgs if p in lat_set), None)
                    or imgs[0]
                )

                abs_path = files_root / chosen
                if not abs_path.exists():
                    # Older mirrors may store files at the data_root level instead.
                    alt = data_root / chosen
                    if alt.exists():
                        abs_path = alt
                    else:
                        n_skipped_missing_file += 1
                        continue

                rows_out.append({
                    "study_id":   sid.lstrip("s"),     # drop the 's' prefix for cleaner ids
                    "subject_id": subject_id,
                    "image_path": str(abs_path),
                    "text":       report,
                })

    out_df = pd.DataFrame(rows_out).drop_duplicates(subset=["study_id"])
    ensure_dir(out_reports_csv.parent)
    out_df.to_csv(out_reports_csv, index=False)

    log.info("Wrote %d unique studies -> %s", len(out_df), out_reports_csv)
    log.info("Skipped: %d rows w/o matching text, %d rows w/ missing image file",
             n_skipped_missing_text, n_skipped_missing_file)
    return len(out_df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kaggle.yaml")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip the validate CSV (use train only).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    preprocess(
        data_root        = paths.data_root,
        out_reports_csv  = paths.reports_csv,
        include_validate = not args.no_validate,
    )


if __name__ == "__main__":
    main()
