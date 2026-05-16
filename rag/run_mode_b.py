"""Mode B: answer every question in the VQA test split, for both systems.

Outputs:
  outputs/mode_b_rag.jsonl
  outputs/mode_b_baseline.jsonl

Each line:
  {question_id, study_id, image_path, question, question_type,
   reference, prediction, system, retrieved: [...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_prep.common import Paths, ensure_dir, get_logger, load_config
from generation.medgemma_wrapper import GenConfig, MedGemma
from rag.pipeline import Pipeline, PipelineConfig
from retrieval.colpali_search import ColPaliSearcher

log = get_logger("run_mode_b")


def _build_pipeline(cfg: dict, paths: Paths, system: str) -> Pipeline:
    gen = MedGemma(
        model_name=cfg["models"]["medgemma"],
        device=cfg["generation"]["device"],
        use_4bit=cfg["generation"]["use_4bit"],
    )
    searcher = None
    if system == "rag":
        index_path = paths.artifacts_dir / "colpali_index.pt"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found. Run retrieval.colpali_index first."
            )
        searcher = ColPaliSearcher(
            index_path,
            device=cfg["generation"]["device"],
            use_4bit=cfg["retrieval"]["use_4bit"],
        )
    pcfg = PipelineConfig(
        system  = system,
        top_k   = cfg["retrieval"]["top_k"],
        gen_vqa = GenConfig(
            max_new_tokens = cfg["generation"]["max_new_tokens"]["vqa"],
            temperature    = cfg["generation"]["temperature"],
            do_sample      = cfg["generation"]["do_sample"],
            num_beams      = cfg["generation"]["num_beams"],
        ),
    )
    return Pipeline(generator=gen, searcher=searcher, cfg=pcfg)


def run(cfg: dict, paths: Paths, system: str) -> Path:
    vqa_test = paths.splits_dir / "vqa_test.parquet"
    if not vqa_test.exists():
        raise FileNotFoundError(
            f"{vqa_test} missing — run data_prep.build_vqa_dataset then split_vqa first."
        )
    df = pd.read_parquet(vqa_test)
    pipe = _build_pipeline(cfg, paths, system)

    out_path = paths.outputs_dir / f"mode_b_{system}.jsonl"
    ensure_dir(out_path.parent)
    if out_path.exists():
        out_path.unlink()

    n_kept = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Mode B · {system}"):
        try:
            r = pipe.answer_question(row["image_path"], row["question"])
        except Exception as e:
            log.warning("Skipped %s: %s", row["question_id"], e)
            continue
        rec = {
            "question_id"  : str(row["question_id"]),
            "study_id"     : str(row["study_id"]),
            "subject_id"   : str(row.get("subject_id", "")),
            "image_path"   : str(row["image_path"]),
            "question"     : row["question"],
            "question_type": row["question_type"],
            "reference"    : row["answer"],
            "prediction"   : r["prediction"],
            "system"       : r["system"],
            "retrieved"    : r["retrieved"],
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_kept += 1

    log.info("Wrote %d rows -> %s", n_kept, out_path)
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
        run(cfg, paths, s)


if __name__ == "__main__":
    main()
