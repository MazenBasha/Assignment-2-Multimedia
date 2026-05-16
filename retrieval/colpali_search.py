"""Late-interaction MaxSim search against a ColPali index.

A query image is embedded to (Q_q, D). Each indexed image has (Q_i, D).
The MaxSim score is the standard ColBERT-style:

    score(q, i) = sum_{t in Q_q} max_{s in Q_i} <q_t, i_s>

We compute it batched in fp16 on GPU when available, falling back to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


@dataclass
class Retrieved:
    rank: int
    score: float
    study_id: str
    subject_id: str
    image_path: str
    text: str


class ColPaliSearcher:
    def __init__(self, index_path: str | Path, *, device: str = "cuda", use_4bit: bool = False):
        from colpali_engine.models import ColPali, ColPaliProcessor

        blob = torch.load(index_path, map_location="cpu", weights_only=False)
        self.embeddings: list[torch.Tensor] = blob["embeddings"]
        self.metadata: list[dict] = blob["metadata"]
        self.model_name: str = blob["model_name"]

        self.device = (
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        if use_4bit:
            from transformers import BitsAndBytesConfig
            self.model = ColPali.from_pretrained(
                self.model_name,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                      bnb_4bit_compute_dtype=torch.float16),
                device_map="auto",
            ).eval()
        else:
            self.model = ColPali.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map=self.device,
            ).eval()
        self.processor = ColPaliProcessor.from_pretrained(self.model_name)

        # Pre-stack the index embeddings into a single (N, Q, D) fp16 tensor
        # on the search device. All ColPali image embeddings share Q for a
        # fixed input resolution; if they vary, we pad with zeros which the
        # max() over inner product will simply ignore.
        Q = max(e.shape[0] for e in self.embeddings)
        D = self.embeddings[0].shape[1]
        N = len(self.embeddings)
        bank = torch.zeros((N, Q, D), dtype=torch.float16)
        for i, e in enumerate(self.embeddings):
            bank[i, : e.shape[0]] = e
        self.bank = bank.to(self.device)

    @torch.inference_mode()
    def _embed_query(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor.process_images([image.convert("RGB")]).to(self.model.device)
        emb = self.model(**inputs)[0]                        # (Q_q, D)
        return emb.to(self.device, dtype=torch.float16)

    @torch.inference_mode()
    def search(self, image: Image.Image, top_k: int = 5) -> list[Retrieved]:
        q = self._embed_query(image)                         # (Qq, D)
        # bank: (N, Qi, D)
        # sim:  (N, Qq, Qi)
        sim = torch.einsum("td,nsd->nts", q, self.bank)
        maxsim = sim.max(dim=-1).values.sum(dim=-1)          # (N,)
        topk = torch.topk(maxsim, k=min(top_k, maxsim.shape[0]))
        out: list[Retrieved] = []
        for rank, (score, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
            m = self.metadata[idx]
            out.append(Retrieved(
                rank       = rank,
                score      = float(score),
                study_id   = str(m["study_id"]),
                subject_id = str(m.get("subject_id", "")),
                image_path = str(m["image_path"]),
                text       = str(m["text"]),
            ))
        return out

    @torch.inference_mode()
    def search_path(self, image_path: str | Path, top_k: int = 5) -> list[Retrieved]:
        return self.search(Image.open(image_path), top_k=top_k)


def format_retrieved_for_prompt(items: Sequence[Retrieved]) -> str:
    """Render retrieved reports into the block used by RAG prompts."""
    lines: list[str] = []
    for r in items:
        lines.append(f"[#{r.rank + 1}  similarity={r.score:.3f}  study_id={r.study_id}]")
        lines.append(r.text.strip())
        lines.append("")
    return "\n".join(lines).strip()
