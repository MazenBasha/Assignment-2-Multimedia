"""Shared eval utilities: tokenisation, JSONL IO, comparison table."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def read_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def f1_token(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    common: dict[str, int] = {}
    for t in p:
        common[t] = common.get(t, 0)
    pcount: dict[str, int] = {}
    rcount: dict[str, int] = {}
    for t in p:
        pcount[t] = pcount.get(t, 0) + 1
    for t in r:
        rcount[t] = rcount.get(t, 0) + 1
    overlap = 0
    for t, c in pcount.items():
        overlap += min(c, rcount.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pcount.values())
    recall    = overlap / sum(rcount.values())
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return float(tokenize(pred) == tokenize(ref))
