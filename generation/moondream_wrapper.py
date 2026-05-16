"""Moondream2 wrapper -- local substitute for MedGemma on a 4 GB GPU / CPU.

Moondream2 (vikhyatk/moondream2, Apache 2.0) has its own simple interface:
    model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
    image_emb = model.encode_image(image)
    answer = model.answer_question(image_emb, question, tokenizer)

We expose the same `MedGemma`-shaped `.generate(image, system, user, cfg)`
method so the RAG pipeline code is identical regardless of generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image


@dataclass
class GenConfig:
    max_new_tokens: int = 180
    temperature:    float = 0.2
    do_sample:      bool = False
    num_beams:      int = 1


class Moondream:
    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        device: str = "cpu",
        use_4bit: bool = False,             # unused on CPU
        dtype: torch.dtype = torch.float32,  # CPU does not support bf16/fp16 reliably
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        kwargs = {"trust_remote_code": True, "revision": "2024-07-23"}
        if device == "cuda":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"]  = "auto"
        else:
            kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        if device == "cpu":
            self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=kwargs["revision"])
        self.model_name = model_name

    def _fold_system_into_question(self, system: str, user: str) -> str:
        """Moondream2 doesn't take separate system/user roles. Fuse them."""
        system = (system or "").strip()
        user   = (user   or "").strip()
        if not system:
            return user
        return f"{system}\n\n{user}"

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        *,
        system: str,
        user: str,
        cfg: Optional[GenConfig] = None,
    ) -> str:
        cfg = cfg or GenConfig()
        prompt = self._fold_system_into_question(system, user)
        enc = self.model.encode_image(image.convert("RGB"))
        # Moondream's answer_question caps tokens internally; we forward our cap.
        ans = self.model.answer_question(
            enc, prompt, self.tokenizer,
            max_new_tokens = cfg.max_new_tokens,
        )
        return (ans or "").strip()


def split_prompt_template(text: str) -> tuple[str, str]:
    """Same helper as in medgemma_wrapper -- split a prompt file on `USER:` line."""
    import re
    parts = re.split(r"^\s*USER:\s*$", text, maxsplit=1, flags=re.MULTILINE)
    if len(parts) != 2:
        raise ValueError("Prompt file must contain SYSTEM: and USER: blocks")
    system = parts[0].replace("SYSTEM:", "", 1).strip()
    user   = parts[1].strip()
    return system, user
