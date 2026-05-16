"""Rule-based VQA pair extraction from radiology report sections.

No LLM judge, no API key. Lower quality than an LLM-generated set, but
lets Mode B run on a laptop without internet. The output schema is
identical to `build_vqa_dataset.py` so `split_vqa.py` and `run_mode_b.py`
need no changes.

Heuristics:
  - For each canonical CheXpert finding, look for affirmative or negated
    mentions in FINDINGS+IMPRESSION; emit a `presence` Q.
  - If an affirmative mention is accompanied by a severity adjective
    (mild/moderate/severe/large/small), emit a `severity` Q.
  - If a finding is associated with a side (right/left/bilateral), emit
    a `location` Q.
  - Always emit a `description` Q whose answer is the IMPRESSION text.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .common import Paths, ensure_dir, get_logger, load_config
from .report_sections import parse as parse_report

log = get_logger("build_vqa_local")


FINDINGS = {
    "pleural effusion":   r"pleural\s+effusion(?:s)?",
    "pneumothorax":       r"pneumothorax",
    "consolidation":      r"consolidation",
    "atelectasis":        r"atelectasis",
    "edema":              r"(?:pulmonary\s+)?edema",
    "cardiomegaly":       r"cardiomegaly|enlarged\s+(?:cardiac|heart)\s+silhouette",
    "pneumonia":          r"pneumonia",
    "lung opacity":       r"(?:lung\s+)?opacit(?:y|ies)",
    "fracture":           r"fracture",
}

NEGATION_PATTERN = re.compile(
    r"\b(?:no|without|absence\s+of|negative\s+for|no\s+evidence\s+of)\b\s+(?:[a-zA-Z- ]+?\s+)?",
    re.IGNORECASE,
)
SEVERITY_WORDS = ["mild", "moderate", "severe", "large", "small", "trace", "minimal"]
SIDE_WORDS     = ["right", "left", "bilateral"]


def _qid(study_id: str, q: str) -> str:
    return "qa_" + hashlib.sha1(f"{study_id}|{q}".encode("utf-8")).hexdigest()[:12]


def _has_negation_near(text: str, finding_match: re.Match) -> bool:
    """True if a negation keyword appears within ~40 chars BEFORE the finding."""
    start = max(0, finding_match.start() - 40)
    window = text[start:finding_match.start() + 1]
    return bool(NEGATION_PATTERN.search(window))


def _severity_near(text: str, m: re.Match) -> str | None:
    start = max(0, m.start() - 30)
    window = text[start:m.end() + 10].lower()
    for w in SEVERITY_WORDS:
        if w in window:
            return w
    return None


def _side_near(text: str, m: re.Match) -> str | None:
    start = max(0, m.start() - 30)
    window = text[start:m.end() + 10].lower()
    for s in SIDE_WORDS:
        if s in window:
            return s
    return None


def _pairs_for_report(study_id: str, sections, n_max: int) -> list[dict]:
    text = (sections.findings + " " + sections.impression).strip()
    if not text:
        return []
    text_lower = text.lower()
    pairs: list[dict] = []

    for finding_name, pat in FINDINGS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            negated = _has_negation_near(text, m)
            q = f"Is there {finding_name}?"
            a = "No." if negated else f"Yes, {finding_name}."
            rationale = text[max(0, m.start()-15): m.end()+15].strip()
            pairs.append({
                "question":      q,
                "answer":        a,
                "question_type": "presence",
                "rationale":     rationale,
            })

            if not negated:
                sev = _severity_near(text, m)
                if sev:
                    pairs.append({
                        "question":      f"How severe is the {finding_name}?",
                        "answer":        sev.capitalize() + ".",
                        "question_type": "severity",
                        "rationale":     rationale,
                    })
                side = _side_near(text, m)
                if side:
                    pairs.append({
                        "question":      f"Which side has the {finding_name}?",
                        "answer":        side.capitalize() + ".",
                        "question_type": "location",
                        "rationale":     rationale,
                    })
            break  # one Q per finding per report

    if sections.impression:
        pairs.append({
            "question":      "Describe the impression of this chest X-ray.",
            "answer":        sections.impression.strip(),
            "question_type": "description",
            "rationale":     sections.impression.strip(),
        })

    # Deduplicate by question text + cap.
    seen: set[str] = set()
    out: list[dict] = []
    for p in pairs:
        if p["question"] in seen:
            continue
        seen.add(p["question"])
        out.append(p)
        if len(out) >= n_max:
            break
    return out


def build(reports_csv: Path, out_parquet: Path, n_pairs_per_report: int) -> int:
    df = pd.read_csv(reports_csv).drop_duplicates(subset=["study_id"])
    rows: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="rule-based VQA"):
        sections = parse_report(str(row.get("text") or ""))
        for p in _pairs_for_report(str(row["study_id"]), sections, n_pairs_per_report):
            rows.append({
                "question_id":   _qid(str(row["study_id"]), p["question"]),
                "study_id":      str(row["study_id"]),
                "subject_id":    str(row.get("subject_id", "")),
                "image_path":    str(row["image_path"]),
                "question":      p["question"],
                "answer":        p["answer"],
                "question_type": p["question_type"],
                "rationale":     p["rationale"],
                "source_report": str(row.get("text") or ""),
            })

    ensure_dir(out_parquet.parent)
    out_df = pd.DataFrame(rows).drop_duplicates(subset=["study_id", "question"])
    out_df.to_parquet(out_parquet, index=False)
    log.info("Wrote %d VQA pairs over %d studies -> %s",
             len(out_df), out_df["study_id"].nunique() if len(out_df) else 0, out_parquet)
    if len(out_df):
        log.info("Type distribution:\n%s", out_df["question_type"].value_counts().to_string())
    return len(out_df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    n_per = cfg["vqa_generation"]["n_pairs_per_report"]
    build(paths.reports_csv, paths.vqa_dataset, n_per)


if __name__ == "__main__":
    main()
