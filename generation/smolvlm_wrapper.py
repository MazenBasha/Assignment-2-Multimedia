"""SmolVLM-256M wrapper.

SmolVLM is very small (~256M params) and gets confused by the long
RAG prompts that work fine for MedGemma. So this wrapper rewrites
the user prompt before sending it to the model:

  * `<image>` placeholders are stripped (the image is attached
    structurally via the chat-template).
  * Long blocks of retrieved reports (formatted by
    `format_retrieved_for_prompt`) are summarised down to a single
    sentence: the FIRST retrieved report's IMPRESSION (or its first
    ~30 words if no IMPRESSION section).
  * A simple, explicit instruction is appended so the model does not
    echo the prompt.

Net effect: the model still benefits from retrieval, but only one
short snippet of context survives -- which is what a 256M model can
realistically use.
"""

from __future__ import annotations

import re
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


_RETRIEVED_BLOCK_RE = re.compile(
    r"\[#\d+\s+similarity=[\d.]+\s+study_id=[^\]]+\]\s*(.+?)(?=\n\s*\[#\d+|\Z)",
    re.DOTALL,
)
_IMPRESSION_RE = re.compile(r"impression[:.]\s*(.+?)(?:\n\s*\n|\Z)",
                            re.IGNORECASE | re.DOTALL)


def _extract_summary(user_text: str) -> str | None:
    """Pull the first retrieved report's IMPRESSION line (or a short snippet)
    out of the RAG-prompt block; return None for baseline prompts that have
    no retrieved-reports block.
    """
    m = _RETRIEVED_BLOCK_RE.search(user_text)
    if not m:
        return None
    body = m.group(1).strip()
    imp = _IMPRESSION_RE.search(body)
    if imp:
        snip = imp.group(1).strip()
    else:
        snip = body
    snip = " ".join(snip.split())               # collapse whitespace
    if len(snip) > 220:
        snip = snip[:217].rstrip() + "..."
    return snip


def _simplify_user_prompt(user: str, *, is_vqa: bool) -> str:
    """Strip <image> tokens, fold retrieved reports down to one sentence, and
    append an explicit instruction. Works on both RAG and baseline prompts."""
    user = user.replace("<image>", "")
    summary = _extract_summary(user)

    if is_vqa:
        # Extract the question, if any.
        qmatch = re.search(r"QUESTION:\s*(.+?)\n", user, re.DOTALL)
        question = qmatch.group(1).strip() if qmatch else user.strip()
        if summary:
            return (f"Reference note from a similar chest X-ray: \"{summary}\".\n"
                    f"Question about THIS chest X-ray: {question}\n"
                    f"Answer briefly. Do not list the reference, do not say "
                    f"\"see reference\". Give only the answer.")
        return f"Question about this chest X-ray: {question}\nAnswer briefly."

    # Mode A (report generation)
    if summary:
        return (f"Reference note from a similar chest X-ray: \"{summary}\".\n"
                f"Write a brief radiology report for THIS chest X-ray. "
                f"Include FINDINGS and IMPRESSION. Do not list reference notes, "
                f"do not say \"see reference\".")
    return ("Write a brief radiology report for this chest X-ray. "
            "Include FINDINGS and IMPRESSION.")


class SmolVLM:
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        device: str = "cpu",
        use_4bit: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_name)
        kwargs = {"torch_dtype": dtype}
        if device == "cuda":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, **kwargs).eval()
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model_name = model_name

    def _build_messages(self, system: str, user: str) -> list[dict]:
        is_vqa = "QUESTION:" in user
        compact_user = _simplify_user_prompt(user, is_vqa=is_vqa)
        compact_sys  = "You are a careful radiologist. Look at the chest X-ray and answer concisely."
        return [
            {"role": "system", "content": [{"type": "text", "text": compact_sys}]},
            {"role": "user",   "content": [
                {"type": "image"},
                {"type": "text", "text": compact_user},
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
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[prompt],
                                images=[image.convert("RGB")],
                                return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens = cfg.max_new_tokens,
            do_sample      = cfg.do_sample,
            temperature    = cfg.temperature if cfg.do_sample else None,
            num_beams      = cfg.num_beams,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
