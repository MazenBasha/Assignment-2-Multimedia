"""Mode A: generate a radiology report for every X-ray in the test split,
for both systems (ColPali-RAG and MedGemma-only baseline).

Writes one JSONL per system to `outputs/`:
  - outputs/mode_a_rag.jsonl
  - outputs/mode_a_baseline.jsonl

Each line:
  {study_id, image_path, reference, prediction, system, retrieved: [...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_prep.common import Paths, ensure_dir, get_logger, load_config, write_jsonl
from data_prep.report_sections import reference_text_for_eval
from generation.medgemma_wrapper import GenConfig
from rag.factory import make_generator, make_searcher
from rag.pipeline import Pipeline, PipelineConfig

log = get_logger("run_mode_a")


def _build_pipeline(cfg: dict, paths: Paths, system: str) -> Pipeline:
    gen = make_generator(cfg)
    searcher = make_searcher(cfg, paths.artifacts_dir) if system == "rag" else None
    pcfg = PipelineConfig(
        system     = system,
        top_k      = cfg["retrieval"]["top_k"],
        gen_report = GenConfig(
            max_new_tokens = cfg["generation"]["max_new_tokens"]["report"],
            temperature    = cfg["generation"]["temperature"],
            do_sample      = cfg["generation"]["do_sample"],
            num_beams      = cfg["generation"]["num_beams"],
        ),
    )
    return Pipeline(generator=gen, searcher=searcher, cfg=pcfg)


def run(cfg: dict, paths: Paths, system: str) -> Path:
    test_parquet = paths.splits_dir / "test.parquet"
    if not test_parquet.exists():
        raise FileNotFoundError(f"{test_parquet} missing — run data_prep.split first.")
    df = pd.read_parquet(test_parquet)

    pipe = _build_pipeline(cfg, paths, system)

    out_path = paths.outputs_dir / f"mode_a_{system}.jsonl"
    ensure_dir(out_path.parent)
    rows: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Mode A · {system}"):
        try:
            r = pipe.generate_report(row["image_path"])
        except Exception as e:
            log.warning("Skipped %s: %s", row["image_path"], e)
            continue
        rows.append({
            "study_id"  : str(row["study_id"]),
            "subject_id": str(row.get("subject_id", "")),
            "image_path": str(row["image_path"]),
            "reference" : reference_text_for_eval(str(row["text"])),
            "prediction": r["prediction"],
            "system"    : r["system"],
            "retrieved" : r["retrieved"],
        })
        # Stream-write so a crash doesn't lose progress.
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
    log.info("Wrote %d rows -> %s", len(rows), out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--system", choices=["rag", "baseline", "both"], default="both")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    systems = ["rag", "baseline"] if args.system == "both" else [args.system]
    for s in systems:
        # Truncate the output file before this run so reruns are clean.
        out_path = paths.outputs_dir / f"mode_a_{s}.jsonl"
        if out_path.exists():
            out_path.unlink()
        run(cfg, paths, s)


if __name__ == "__main__":
    main()
