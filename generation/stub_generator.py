"""A trivial nearest-neighbour generator -- useful as both
  (a) a CPU-cheap end-to-end smoke test (no model weights to download), and
  (b) a real baseline: how good is "copy the nearest retrieved report"?

For Mode A (report): returns the highest-similarity retrieved report,
optionally truncated. The pipeline does the retrieval; this generator
just sees the formatted prompt and pulls the first report out of it.

For Mode B (VQA): rule-based answer from the question keywords + the
top retrieved report. Looks for the same canonical CheXpert findings
the local VQA builder uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class GenConfig:
    max_new_tokens: int = 180
    temperature:    float = 0.2
    do_sample:      bool = False
    num_beams:      int = 1


_FINDINGS_KEYWORDS = [
    "pleural effusion", "pneumothorax", "consolidation", "atelectasis",
    "edema", "cardiomegaly", "pneumonia", "opacity", "fracture",
]
_NEGATION_RE = re.compile(r"\b(?:no|without|absence of|negative for|no evidence of)\b", re.IGNORECASE)


def _first_retrieved_report(user_prompt: str) -> str:
    """Yank the first retrieved report from the formatted RAG prompt.
    The block format is the one emitted by `format_retrieved_for_prompt`:
        [#1  similarity=...  study_id=...]\n<text>\n\n[#2 ...]
    """
    blocks = re.split(r"\n\s*\[#\d+\s+similarity=", user_prompt)
    if len(blocks) < 2:
        return ""
    body = blocks[1].split("\n", 1)
    return body[1].strip() if len(body) == 2 else body[0].strip()


def _classify_question(q: str) -> tuple[str, str | None]:
    """Return (question_type, target_finding) using crude keyword rules."""
    ql = q.lower()
    for f in _FINDINGS_KEYWORDS:
        if f in ql:
            if any(w in ql for w in ("is there", "any", "presence of")):
                return "presence", f
            if any(w in ql for w in ("how severe", "severity")):
                return "severity", f
            if any(w in ql for w in ("which side", "where")):
                return "location", f
            return "presence", f
    if "describe" in ql or "impression" in ql:
        return "description", None
    return "presence", None


class StubGenerator:
    """Same .generate(image, system, user, cfg) signature as MedGemma/Moondream."""

    def __init__(self, *args, **kwargs):
        self.model_name = "stub-nearest-neighbour"

    def generate(self, image: Image.Image, *, system: str, user: str,
                 cfg: Optional[GenConfig] = None) -> str:
        cfg = cfg or GenConfig()
        is_vqa = "QUESTION:" in user

        retrieved = _first_retrieved_report(user)

        if not is_vqa:
            # Mode A: return the nearest retrieved report verbatim (truncated to budget).
            # Token budget is approximate -> use a char cap.
            char_cap = cfg.max_new_tokens * 4
            return (retrieved[:char_cap].strip() if retrieved
                    else "FINDINGS: No abnormality identified.\nIMPRESSION: Normal chest radiograph.")

        # Mode B: parse the question, decide from the retrieved report.
        m = re.search(r"QUESTION:\s*(.+?)\n", user, re.DOTALL)
        q = (m.group(1).strip() if m else "").lower()
        qtype, finding = _classify_question(q)
        rl = retrieved.lower()

        if qtype == "presence":
            if finding and finding in rl:
                # check for negation in a 60-char window around the mention
                idx = rl.find(finding)
                window = rl[max(0, idx-40): idx+10]
                if _NEGATION_RE.search(window):
                    return "No."
                return f"Yes, {finding}."
            return "No."

        if qtype == "severity":
            for w in ("mild", "moderate", "severe", "large", "small", "trace", "minimal"):
                if w in rl:
                    return w.capitalize() + "."
            return "Not specified."

        if qtype == "location":
            for w in ("right", "left", "bilateral"):
                if w in rl:
                    return w.capitalize() + "."
            return "Not specified."

        if qtype == "description":
            # Try to extract IMPRESSION section
            m = re.search(r"impression:\s*(.+)$", retrieved, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip().split("\n\n")[0][:200]
            return retrieved[:200] if retrieved else "Normal chest radiograph."

        return "Cannot be determined from the image."
