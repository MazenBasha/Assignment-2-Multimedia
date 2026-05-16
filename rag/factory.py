"""Build the (generator, retriever) pair from a config dict.

Lets `run_mode_a.py`, `run_mode_b.py`, and any future driver script swap
between the assignment-brief stack (ColPali + MedGemma) and the laptop
stack (CLIP + Moondream2) by changing two config keys:

    retrieval.backend  : "colpali" | "clip"
    generation.backend : "medgemma" | "moondream"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def make_generator(cfg: dict):
    backend = cfg["generation"].get("backend", "medgemma")
    device  = cfg["generation"].get("device",  "cuda")
    use_4bit = cfg["generation"].get("use_4bit", False)

    if backend == "medgemma":
        from generation.medgemma_wrapper import MedGemma
        return MedGemma(
            model_name = cfg["models"]["medgemma"],
            device     = device,
            use_4bit   = use_4bit,
        )
    if backend == "moondream":
        from generation.moondream_wrapper import Moondream
        return Moondream(
            model_name = cfg["generation"].get("moondream_model", "vikhyatk/moondream2"),
            device     = device,
            use_4bit   = use_4bit,
        )
    raise ValueError(f"Unknown generation.backend: {backend}")


def make_searcher(cfg: dict, artifacts_dir: Path) -> Optional[Any]:
    backend = cfg["retrieval"].get("backend", "colpali")
    device  = cfg["generation"].get("device",  "cuda")
    use_4bit = cfg["retrieval"].get("use_4bit", False)

    if backend == "colpali":
        index_path = artifacts_dir / "colpali_index.pt"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found. Run retrieval.colpali_index first."
            )
        from retrieval.colpali_search import ColPaliSearcher
        return ColPaliSearcher(index_path, device=device, use_4bit=use_4bit)
    if backend == "clip":
        index_path = artifacts_dir / "clip_index.pt"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found. Run retrieval.clip_index first."
            )
        from retrieval.clip_search import CLIPSearcher
        return CLIPSearcher(index_path, device=device, use_4bit=use_4bit)
    raise ValueError(f"Unknown retrieval.backend: {backend}")


def make_index_command(cfg: dict) -> list[str]:
    """Return the module path that builds the retrieval index for this config."""
    backend = cfg["retrieval"].get("backend", "colpali")
    if backend == "colpali":
        return ["retrieval.colpali_index"]
    if backend == "clip":
        return ["retrieval.clip_index"]
    raise ValueError(f"Unknown retrieval.backend: {backend}")
