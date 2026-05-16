"""Cherry-pick qualitative side-by-side examples for the report and demo.

For Mode A:
  - 5 examples per system where RAG beats baseline on BERTScore
  - 5 where baseline beats RAG (failure analysis)

For Mode B:
  - 5 per question_type, RAG vs. baseline

Output:
    outputs/qualitative_mode_a.md
    outputs/qualitative_mode_b.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_prep.common import Paths, ensure_dir, get_logger, load_config
from .common import f1_token, read_jsonl

log = get_logger("qualitative")


def _mode_a(paths: Paths, top_n: int = 5) -> None:
    rag  = {r["study_id"]: r for r in read_jsonl(paths.outputs_dir / "mode_a_rag.jsonl")}
    base = {r["study_id"]: r for r in read_jsonl(paths.outputs_dir / "mode_a_baseline.jsonl")}
    shared = sorted(set(rag) & set(base))
    if not shared:
        log.warning("No overlap between RAG and baseline Mode A outputs.")
        return

    diffs = []
    for sid in shared:
        ref = rag[sid]["reference"]
        diff = f1_token(rag[sid]["prediction"], ref) - f1_token(base[sid]["prediction"], ref)
        diffs.append((diff, sid))
    diffs.sort(reverse=True)

    wins   = diffs[:top_n]
    losses = diffs[-top_n:][::-1]

    out_path = paths.outputs_dir / "qualitative_mode_a.md"
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Mode A — qualitative examples\n\n")
        for header, group in (("RAG wins", wins), ("RAG losses", losses)):
            f.write(f"## {header}\n\n")
            for delta, sid in group:
                f.write(f"### study_id={sid}  Δtoken_F1={delta:+.3f}\n\n")
                f.write(f"**Image**: `{rag[sid]['image_path']}`\n\n")
                f.write("**Reference**:\n\n```\n" + rag[sid]["reference"] + "\n```\n\n")
                f.write("**RAG prediction**:\n\n```\n" + rag[sid]["prediction"] + "\n```\n\n")
                f.write("**Baseline prediction**:\n\n```\n" + base[sid]["prediction"] + "\n```\n\n")
                if rag[sid].get("retrieved"):
                    f.write("**Top retrieved study_ids**: " +
                            ", ".join(r["study_id"] for r in rag[sid]["retrieved"][:3]) + "\n\n")
                f.write("---\n\n")
    log.info("Wrote %s", out_path)


def _mode_b(paths: Paths, per_type: int = 3) -> None:
    rag  = {r["question_id"]: r for r in read_jsonl(paths.outputs_dir / "mode_b_rag.jsonl")}
    base = {r["question_id"]: r for r in read_jsonl(paths.outputs_dir / "mode_b_baseline.jsonl")}
    shared = sorted(set(rag) & set(base))

    out_path = paths.outputs_dir / "qualitative_mode_b.md"
    ensure_dir(out_path.parent)
    by_type: dict[str, list[tuple[float, str]]] = {}
    for qid in shared:
        ref = rag[qid]["reference"]
        delta = f1_token(rag[qid]["prediction"], ref) - f1_token(base[qid]["prediction"], ref)
        by_type.setdefault(rag[qid].get("question_type", "unknown"), []).append((delta, qid))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Mode B — qualitative examples (per question type)\n\n")
        for qtype, items in sorted(by_type.items()):
            items.sort(reverse=True)
            f.write(f"## {qtype} (n={len(items)})\n\n")
            for delta, qid in (items[:per_type] + items[-per_type:]):
                f.write(f"### qid={qid}  Δtoken_F1={delta:+.3f}\n\n")
                f.write(f"- **Q**: {rag[qid]['question']}\n")
                f.write(f"- **A (gold)**: {rag[qid]['reference']}\n")
                f.write(f"- **RAG**:      {rag[qid]['prediction']}\n")
                f.write(f"- **Baseline**: {base[qid]['prediction']}\n\n")
                f.write("---\n\n")
    log.info("Wrote %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    if (paths.outputs_dir / "mode_a_rag.jsonl").exists():
        _mode_a(paths)
    if (paths.outputs_dir / "mode_b_rag.jsonl").exists():
        _mode_b(paths)


if __name__ == "__main__":
    main()
