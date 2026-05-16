"""Mode A metrics: BLEU-1/2/3/4, ROUGE-L, METEOR, BERTScore F1,
and optional CheXbert macro-F1 over the 14 CheXpert findings.

Input  : outputs/mode_a_{system}.jsonl with `reference` and `prediction`.
Output : outputs/metrics_mode_a.json   summary across systems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_prep.common import Paths, ensure_dir, get_logger, load_config
from .common import read_jsonl, tokenize

log = get_logger("metrics_report")


def _bleu(refs: list[str], hyps: list[str], n: int) -> float:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
    refs_tok = [[tokenize(r)] for r in refs]
    hyps_tok = [tokenize(h)   for h in hyps]
    return float(corpus_bleu(refs_tok, hyps_tok, weights=weights,
                             smoothing_function=SmoothingFunction().method1))


def _rouge_l(refs: list[str], hyps: list[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return sum(scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)) / max(1, len(refs))


def _meteor(refs: list[str], hyps: list[str]) -> float:
    import nltk
    for pkg in ("wordnet", "punkt", "omw-1.4"):
        try:
            nltk.data.find(pkg if "/" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.translate.meteor_score import meteor_score
    return sum(meteor_score([tokenize(r)], tokenize(h)) for r, h in zip(refs, hyps)) / max(1, len(refs))


def _bertscore(refs: list[str], hyps: list[str], model_type: str) -> float:
    from bert_score import score
    _, _, F1 = score(hyps, refs, model_type=model_type, lang="en",
                     rescale_with_baseline=False)
    return float(F1.mean().item())


def evaluate(system_jsonl: Path, *, bertscore_model: str,
             include_chexbert: bool, include_bertscore: bool,
             chexbert_labels: list[str]) -> dict:
    rows = read_jsonl(system_jsonl)
    refs = [r["reference"]  for r in rows]
    hyps = [r["prediction"] for r in rows]

    log.info("Evaluating %d rows from %s", len(rows), system_jsonl)
    out = {
        "n":            len(rows),
        "bleu_1":       _bleu(refs, hyps, 1),
        "bleu_2":       _bleu(refs, hyps, 2),
        "bleu_3":       _bleu(refs, hyps, 3),
        "bleu_4":       _bleu(refs, hyps, 4),
        "rouge_l":      _rouge_l(refs, hyps),
        "meteor":       _meteor(refs, hyps),
    }
    if include_bertscore:
        out["bertscore_f1"] = _bertscore(refs, hyps, bertscore_model)
    if include_chexbert:
        try:
            from .chexbert_eval import macro_f1_over_findings
            cf1, per = macro_f1_over_findings(refs, hyps, labels=chexbert_labels)
            out["chexbert_macro_f1"] = cf1
            out["chexbert_per_label"] = per
        except Exception as e:
            log.warning("CheXbert evaluation skipped: %s", e)
            out["chexbert_macro_f1"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--skip-chexbert", action="store_true",
                    help="Skip CheXbert evaluation (slow; needs a GPU + weights).")
    ap.add_argument("--skip-bertscore", action="store_true",
                    help="Skip BERTScore evaluation (downloads a large model; needs GPU to be fast).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    results: dict[str, dict] = {}
    for system in ("rag", "baseline"):
        path = paths.outputs_dir / f"mode_a_{system}.jsonl"
        if not path.exists():
            log.warning("Missing %s; run rag.run_mode_a first.", path)
            continue
        results[system] = evaluate(
            path,
            bertscore_model   = cfg["models"]["bertscore"],
            include_chexbert  = not args.skip_chexbert,
            include_bertscore = not args.skip_bertscore,
            chexbert_labels   = cfg["eval"]["chexpert_labels"],
        )

    if "rag" in results and "baseline" in results:
        delta = {}
        for k, v in results["rag"].items():
            if isinstance(v, (int, float)) and isinstance(results["baseline"].get(k), (int, float)):
                delta[k] = v - results["baseline"][k]
        results["delta_rag_minus_baseline"] = delta

    out_path = paths.outputs_dir / "metrics_mode_a.json"
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_path)
    log.info("Summary:\n%s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
