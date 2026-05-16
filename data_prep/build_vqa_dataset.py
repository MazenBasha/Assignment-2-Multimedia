"""Build a VQA dataset on top of the MIMIC-CXR Kaggle mirror.

We follow the recipe sketched by the MIMIC-CXR-VQA paper (Bae et al., 2023)
and the LightVED-prhlt notebook: prompt an instruction-tuned LLM with the
parsed findings/impression of each report and ask it to write N clinically
relevant Q&A pairs whose answers are supported by quoted spans from the
report.

Output:
    data/vqa/mimic_vqa.parquet
    Columns: question_id, study_id, subject_id, image_path,
             question, answer, question_type, rationale, source_report

Filtering steps applied here:
    1. JSON-parse filter — drop completions that aren't a JSON array.
    2. Rationale-substring filter — every `rationale` must be a substring
       of the source report (case-insensitive). This kills hallucinations.
    3. Length filter — drop overly long questions/answers.
    4. Deduplication — identical (study_id, question) collapsed.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .common import Paths, ensure_dir, get_logger, load_config
from .llm_judge import make_judge, parse_strict_json_array
from .report_sections import parse as parse_report

log = get_logger("build_vqa")

QUESTION_TYPES = {"presence", "location", "severity", "differential", "description"}
PROMPT_FILE = Path("prompts/vqa_generation.txt")


def _load_prompt() -> tuple[str, str]:
    """Return (system, user_template) split on the first `USER:` line."""
    txt = PROMPT_FILE.read_text(encoding="utf-8")
    # The file starts with "SYSTEM:" then "USER:". Split on a line that is
    # exactly "USER:".
    parts = re.split(r"^\s*USER:\s*$", txt, maxsplit=1, flags=re.MULTILINE)
    if len(parts) != 2:
        raise RuntimeError(f"{PROMPT_FILE} must contain SYSTEM: and USER: blocks")
    system = parts[0].replace("SYSTEM:", "", 1).strip()
    user   = parts[1].strip()
    return system, user


def _qid(study_id: str, question: str) -> str:
    h = hashlib.sha1(f"{study_id}|{question}".encode("utf-8")).hexdigest()[:12]
    return f"qa_{h}"


def _valid_pair(pair: dict, source_report: str) -> bool:
    if not isinstance(pair, dict):
        return False
    q = (pair.get("question") or "").strip()
    a = (pair.get("answer")   or "").strip()
    qt = (pair.get("question_type") or "").strip().lower()
    rationale = (pair.get("rationale") or "").strip()

    if not q or not a or qt not in QUESTION_TYPES:
        return False
    if len(q) > 300 or len(a) > 300:
        return False
    if rationale and rationale.lower() not in source_report.lower():
        # Rationale must be grounded — otherwise the model hallucinated.
        return False
    return True


def build(
    reports_csv: Path,
    out_parquet: Path,
    judge_provider: str,
    judge_model: str,
    n_pairs_per_report: int,
    n_reports_to_use: int,
    seed: int,
) -> None:
    # Use the full reports.csv (train + test) so that downstream
    # `split_vqa.py` can produce both train and test VQA partitions
    # off the same patient-level split as the report-generation task.
    df_in = pd.read_csv(reports_csv).drop_duplicates(subset=["study_id"])
    if n_reports_to_use and n_reports_to_use < len(df_in):
        df_in = df_in.sample(n=n_reports_to_use, random_state=seed).reset_index(drop=True)
    log.info("Generating VQA over %d reports", len(df_in))

    judge = make_judge(judge_provider, judge_model)
    system, user_tpl = _load_prompt()

    rows: list[dict] = []
    n_dropped_json = 0
    n_dropped_filter = 0

    for _, row in tqdm(df_in.iterrows(), total=len(df_in)):
        study_id = str(row["study_id"])
        sections = parse_report(str(row["text"]))
        if not sections.has_clinical_content():
            continue
        source_text = sections.for_vqa()

        user_prompt = user_tpl.format(
            n_pairs   = n_pairs_per_report,
            report_id = study_id,
            report_text = source_text,
        )
        try:
            raw = judge.complete(system, user_prompt, max_tokens=1024, temperature=0.2)
        except Exception as e:
            log.warning("Judge failed on %s: %s", study_id, e)
            continue

        pairs = parse_strict_json_array(raw)
        if not pairs:
            n_dropped_json += 1
            continue

        for p in pairs:
            if not _valid_pair(p, source_text):
                n_dropped_filter += 1
                continue
            rows.append({
                "question_id"   : _qid(study_id, p["question"].strip()),
                "study_id"      : study_id,
                "subject_id"    : str(row.get("subject_id", "")),
                "image_path"    : str(row["image_path"]),
                "question"      : p["question"].strip(),
                "answer"        : p["answer"].strip(),
                "question_type" : p["question_type"].strip().lower(),
                "rationale"     : (p.get("rationale") or "").strip(),
                "source_report" : source_text,
            })

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["study_id", "question"], keep="first",
    )
    ensure_dir(out_parquet.parent)
    df.to_parquet(out_parquet, index=False)

    log.info("VQA dataset: %d pairs over %d studies", len(df), df["study_id"].nunique())
    log.info("Dropped (JSON parse): %d, (validation filter): %d",
             n_dropped_json, n_dropped_filter)
    if len(df):
        log.info("Type distribution:\n%s", df["question_type"].value_counts().to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    vqa = cfg["vqa_generation"]
    build(
        reports_csv        = paths.reports_csv,
        out_parquet        = paths.vqa_dataset,
        judge_provider     = vqa["judge_provider"],
        judge_model        = vqa["judge_model"],
        n_pairs_per_report = vqa["n_pairs_per_report"],
        n_reports_to_use   = vqa["n_reports_to_use"],
        seed               = cfg["dataset"]["seed"],
    )


if __name__ == "__main__":
    main()
