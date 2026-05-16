"""Clinical scoring: extract the 14 CheXpert labels from generated and
reference reports and compute macro-F1.

We use the CheXbert model — a BERT classifier fine-tuned by Stanford ML
group on radiologist annotations (Smit et al., EMNLP 2020). The default
checkpoint on Hugging Face (`StanfordAIMI/CheXbert`) outputs per-label
logits in the order of the CheXpert 14 findings.

This module is intentionally optional — `metrics_report.py` calls it in a
try/except so the lexical metrics still run on a machine without GPU or
without the checkpoint downloaded.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


_MODEL = None
_TOKENIZER = None
# CheXpert 14 finding order as exposed by the canonical CheXbert checkpoint.
DEFAULT_FINDINGS: list[str] = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices", "No Finding",
]


def _load(model_name: str = "StanfordAIMI/CheXbert"):
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    if torch.cuda.is_available():
        _MODEL = _MODEL.to("cuda")
    return _MODEL, _TOKENIZER


def _label_matrix(reports: list[str]) -> np.ndarray:
    """Return a (N, 14) binary matrix. CheXbert produces 4-way labels
    (negative, uncertain, positive, blank) — we collapse to binary
    positive ∈ {0,1} for macro-F1 over the 14 findings.
    """
    import torch
    model, tokenizer = _load()
    mat = np.zeros((len(reports), len(DEFAULT_FINDINGS)), dtype=np.int8)

    for i, txt in enumerate(reports):
        if not (txt or "").strip():
            continue
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits          # (1, 14, 4) for canonical CheXbert
        if logits.dim() == 3:
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (14,)
            # class index 2 == positive in canonical CheXbert
            mat[i] = (preds == 2).astype(np.int8)
        elif logits.dim() == 2:
            # Some HF mirrors expose flattened binary heads.
            mat[i] = (logits.squeeze(0).cpu().numpy() > 0).astype(np.int8)
    return mat


def macro_f1_over_findings(
    refs: list[str],
    hyps: list[str],
    *,
    labels: Iterable[str] = DEFAULT_FINDINGS,
) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import f1_score
    yr = _label_matrix(refs)
    yh = _label_matrix(hyps)

    label_list = list(labels)
    # Keep only the columns the user asked for, in their order.
    keep_idx = [DEFAULT_FINDINGS.index(l) for l in label_list if l in DEFAULT_FINDINGS]
    yr = yr[:, keep_idx]
    yh = yh[:, keep_idx]
    used_labels = [DEFAULT_FINDINGS[i] for i in keep_idx]

    per_label: dict[str, float] = {}
    for j, name in enumerate(used_labels):
        per_label[name] = float(f1_score(yr[:, j], yh[:, j], zero_division=0))

    macro = float(np.mean(list(per_label.values()))) if per_label else 0.0
    return macro, per_label
