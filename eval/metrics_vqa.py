"""Mode B metrics: Exact-Match, token-level F1, and per-question-type
accuracy. Operates on `outputs/mode_b_{system}.jsonl`.

Output: outputs/metrics_mode_b.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from data_prep.common import Paths, ensure_dir, get_logger, load_config
from .common import exact_match, f1_token, read_jsonl

log = get_logger("metrics_vqa")


def _normalize_yesno(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return t


def evaluate(jsonl_path: Path) -> dict:
    rows = read_jsonl(jsonl_path)
    if not rows:
        return {"n": 0}

    out: dict = {"n": len(rows)}
    em_scores  = [exact_match(r["prediction"], r["reference"]) for r in rows]
    f1_scores  = [f1_token (r["prediction"], r["reference"]) for r in rows]
    out["exact_match"] = sum(em_scores) / len(em_scores)
    out["token_f1"]    = sum(f1_scores) / len(f1_scores)

    # Per-type breakdown
    by_type_em:  dict[str, list[float]] = defaultdict(list)
    by_type_f1:  dict[str, list[float]] = defaultdict(list)
    for r, em, f1 in zip(rows, em_scores, f1_scores):
        by_type_em[r.get("question_type", "unknown")].append(em)
        by_type_f1[r.get("question_type", "unknown")].append(f1)

    per_type: dict[str, dict] = {}
    for k in sorted(by_type_em):
        n = len(by_type_em[k])
        per_type[k] = {
            "n":           n,
            "exact_match": sum(by_type_em[k]) / n,
            "token_f1":    sum(by_type_f1[k]) / n,
        }

    # Presence-only yes/no accuracy: very informative because it removes
    # surface-form noise.
    presence_rows = [r for r in rows if r.get("question_type") == "presence"]
    if presence_rows:
        acc = sum(
            1 for r in presence_rows
            if _normalize_yesno(r["prediction"]) == _normalize_yesno(r["reference"])
        ) / len(presence_rows)
        out["presence_yesno_accuracy"] = acc

    out["per_question_type"] = per_type
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    results: dict = {}
    for system in ("rag", "baseline"):
        p = paths.outputs_dir / f"mode_b_{system}.jsonl"
        if not p.exists():
            log.warning("Missing %s; run rag.run_mode_b first.", p)
            continue
        results[system] = evaluate(p)

    if "rag" in results and "baseline" in results:
        delta: dict[str, float] = {}
        for k, v in results["rag"].items():
            base = results["baseline"].get(k)
            if isinstance(v, (int, float)) and isinstance(base, (int, float)):
                delta[k] = v - base
        results["delta_rag_minus_baseline"] = delta

    out_path = paths.outputs_dir / "metrics_mode_b.json"
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_path)
    log.info("Summary:\n%s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
