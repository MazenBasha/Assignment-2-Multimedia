"""CLIP image-to-image search, drop-in interface compatible with
`retrieval.colpali_search.ColPaliSearcher` (same `Retrieved` dataclass,
same `format_retrieved_for_prompt` helper).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from .colpali_search import Retrieved, format_retrieved_for_prompt  # reuse


class CLIPSearcher:
    def __init__(self, index_path: str | Path, *, device: str = "cpu", use_4bit: bool = False):
        from transformers import CLIPModel, CLIPProcessor

        blob = torch.load(index_path, map_location="cpu", weights_only=False)
        self.embeddings: torch.Tensor = blob["embeddings"]                 # (N, D)
        self.metadata:   list[dict]   = blob["metadata"]
        self.model_name: str          = blob["model_name"]

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        self.model     = CLIPModel.from_pretrained(self.model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.bank      = self.embeddings.to(self.device)

    @torch.inference_mode()
    def _embed_query(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt").to(self.device)
        f = self.model.get_image_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
        return f[0]                                                        # (D,)

    @torch.inference_mode()
    def search(self, image: Image.Image, top_k: int = 3) -> list[Retrieved]:
        q = self._embed_query(image)                                       # (D,)
        sims = self.bank @ q                                               # (N,)
        topk = torch.topk(sims, k=min(top_k, sims.shape[0]))
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
    def search_path(self, image_path: str | Path, top_k: int = 3) -> list[Retrieved]:
        return self.search(Image.open(image_path), top_k=top_k)


__all__ = ["CLIPSearcher", "Retrieved", "format_retrieved_for_prompt"]
