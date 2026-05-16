"""End-to-end pipelines for Mode A (report) and Mode B (VQA).

Two systems:
  - "rag"     : ColPali → retrieve top-k → MedGemma(image, retrieved_reports)
  - "baseline": MedGemma(image) directly, no retrieval

Both systems share the same generator + generation hyperparameters; the
ONLY difference is the prompt and whether retrieval runs. This isolates
the contribution of retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from generation.medgemma_wrapper import GenConfig, split_prompt_template
from retrieval.colpali_search import Retrieved, format_retrieved_for_prompt


PROMPTS_DIR = Path("prompts")


@dataclass
class PipelineConfig:
    system:        str               # "rag" | "baseline"
    top_k:         int = 5
    gen_report:    GenConfig = None
    gen_vqa:       GenConfig = None

    def __post_init__(self):
        if self.gen_report is None:
            self.gen_report = GenConfig(max_new_tokens=350, do_sample=False)
        if self.gen_vqa is None:
            self.gen_vqa = GenConfig(max_new_tokens=64,  do_sample=False)


class Pipeline:
    def __init__(
        self,
        generator,                          # MedGemma | Moondream — duck typed
        searcher,                           # ColPaliSearcher | CLIPSearcher | None
        cfg: PipelineConfig,
    ):
        if cfg.system == "rag" and searcher is None:
            raise ValueError("system='rag' requires a searcher")
        self.generator = generator
        self.searcher  = searcher
        self.cfg       = cfg

        # Lazy-load the prompt files once.
        self._tpl_rag_report      = (PROMPTS_DIR / "rag_report_generation.txt").read_text(encoding="utf-8")
        self._tpl_rag_vqa         = (PROMPTS_DIR / "rag_vqa.txt").read_text(encoding="utf-8")
        self._tpl_base_report     = (PROMPTS_DIR / "baseline_report_generation.txt").read_text(encoding="utf-8")
        self._tpl_base_vqa        = (PROMPTS_DIR / "baseline_vqa.txt").read_text(encoding="utf-8")

    # -------------------------- Mode A : report -----------------------
    def generate_report(self, image_path: str | Path) -> dict:
        image = Image.open(image_path).convert("RGB")
        retrieved: list[Retrieved] = []

        if self.cfg.system == "rag":
            retrieved = self.searcher.search(image, top_k=self.cfg.top_k)
            system, user_tpl = split_prompt_template(self._tpl_rag_report)
            user = user_tpl.format(
                k = len(retrieved),
                retrieved_reports = format_retrieved_for_prompt(retrieved),
            )
        else:
            system, user_tpl = split_prompt_template(self._tpl_base_report)
            user = user_tpl                                # no placeholders

        text = self.generator.generate(
            image, system=system, user=user, cfg=self.cfg.gen_report,
        )
        return {
            "image_path": str(image_path),
            "system":     self.cfg.system,
            "prediction": text,
            "retrieved":  [r.__dict__ for r in retrieved],
        }

    # --------------------------- Mode B : VQA -------------------------
    def answer_question(self, image_path: str | Path, question: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        retrieved: list[Retrieved] = []

        if self.cfg.system == "rag":
            retrieved = self.searcher.search(image, top_k=self.cfg.top_k)
            system, user_tpl = split_prompt_template(self._tpl_rag_vqa)
            user = user_tpl.format(
                retrieved_reports = format_retrieved_for_prompt(retrieved),
                question = question.strip(),
            )
        else:
            system, user_tpl = split_prompt_template(self._tpl_base_vqa)
            user = user_tpl.format(question=question.strip())

        text = self.generator.generate(
            image, system=system, user=user, cfg=self.cfg.gen_vqa,
        )
        return {
            "image_path": str(image_path),
            "question":   question,
            "system":     self.cfg.system,
            "prediction": text,
            "retrieved":  [r.__dict__ for r in retrieved],
        }
