"""MedGemma multimodal generator wrapper.

Usage:
    from generation.medgemma_wrapper import MedGemma
    gen = MedGemma(model_name="google/medgemma-1.5-4b-it", use_4bit=True)
    text = gen.generate(image=Image.open(...), system="You are a radiologist...",
                        user="Write the report.", max_new_tokens=350)

MedGemma is a Gemma-3 derivative; it uses the standard HF multimodal chat
template (`processor.apply_chat_template`). We construct the conversation
in the format the model card recommends:

    messages = [
      {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
      {"role": "user",   "content": [
          {"type": "image"},                # placeholder, real image goes in via processor
          {"type": "text",  "text": USER},
      ]},
    ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image


@dataclass
class GenConfig:
    max_new_tokens: int = 350
    temperature: float = 0.2
    do_sample: bool = False
    num_beams: int = 1


class MedGemma:
    def __init__(
        self,
        model_name: str = "google/medgemma-1.5-4b-it",
        device: str = "cuda",
        use_4bit: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            BitsAndBytesConfig,
        )

        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        kwargs: dict[str, Any] = {}
        if use_4bit and self.device == "cuda":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = dtype
            kwargs["device_map"]  = self.device

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs).eval()
        self.model_name = model_name

    def _build_messages(self, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user",   "content": [
                {"type": "image"},
                {"type": "text",  "text": user},
            ]},
        ]

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
        messages = self._build_messages(system, user)

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        inputs = self.processor(
            text=prompt,
            images=[image.convert("RGB")],
            return_tensors="pt",
        ).to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens = cfg.max_new_tokens,
            do_sample      = cfg.do_sample,
            temperature    = cfg.temperature if cfg.do_sample else None,
            num_beams      = cfg.num_beams,
        )
        # Slice off the prompt tokens; the chat template assistant turn
        # starts after the input length.
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        return text


def split_prompt_template(text: str) -> tuple[str, str]:
    """Split a `prompts/*.txt` file on the first `USER:` line.

    Returns (system, user_template). Lets us reuse the same files as the
    VQA generator (which already does this).
    """
    import re
    parts = re.split(r"^\s*USER:\s*$", text, maxsplit=1, flags=re.MULTILINE)
    if len(parts) != 2:
        raise ValueError("Prompt file must contain SYSTEM: and USER: blocks")
    system = parts[0].replace("SYSTEM:", "", 1).strip()
    user   = parts[1].strip()
    return system, user
