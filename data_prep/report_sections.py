"""Parse FINDINGS / IMPRESSION / INDICATION sections out of MIMIC-CXR
free-text reports. Robust to the variety of section headers used by MIMIC.

This file is intentionally dependency-free so it can be unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Section headers we accept, in order of preference. Each entry is a
# regex alternation. Patterns are matched case-insensitively at the start
# of a line (after optional whitespace).
SECTION_PATTERNS: dict[str, str] = {
    "indication":  r"(indication|history|clinical(?:\s+history)?|reason\s+for(?:\s+exam(?:ination)?)?)",
    "comparison":  r"(comparison|prior(?:s)?)",
    "technique":   r"(technique|procedure)",
    "findings":    r"(findings|report)",
    "impression":  r"(impression|conclusion|interpretation)",
}

_SECTION_HEADER_RE = re.compile(
    r"^\s*(?P<name>" + "|".join(p for p in SECTION_PATTERNS.values()) + r")\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ReportSections:
    indication: str = ""
    comparison: str = ""
    technique:  str = ""
    findings:   str = ""
    impression: str = ""
    raw:        str = ""

    def has_clinical_content(self) -> bool:
        return bool((self.findings + self.impression).strip())

    def for_vqa(self) -> str:
        """Concatenation used as the source text for VQA generation."""
        parts = []
        if self.indication:
            parts.append(f"INDICATION:\n{self.indication}")
        if self.findings:
            parts.append(f"FINDINGS:\n{self.findings}")
        if self.impression:
            parts.append(f"IMPRESSION:\n{self.impression}")
        return "\n\n".join(parts)


def _canonical_section_name(header_text: str) -> str | None:
    h = header_text.lower()
    for name, pat in SECTION_PATTERNS.items():
        if re.fullmatch(pat, h, re.IGNORECASE):
            return name
    return None


def parse(report: str) -> ReportSections:
    """Split a MIMIC-CXR report into its labelled sections.

    Strategy: locate every line that starts with `KNOWN_HEADER:`, treat the
    text in between as the body of the preceding section.
    """
    if not report or not report.strip():
        return ReportSections(raw=report or "")

    matches = list(_SECTION_HEADER_RE.finditer(report))
    sections: dict[str, str] = {}

    if not matches:
        # No headers — heuristic: dump everything into findings.
        return ReportSections(findings=report.strip(), raw=report)

    for i, m in enumerate(matches):
        canonical = _canonical_section_name(m.group("name"))
        if canonical is None:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(report)
        body = report[start:end].strip()
        # Some reports duplicate a header; keep the longest body.
        if len(body) > len(sections.get(canonical, "")):
            sections[canonical] = body

    return ReportSections(
        indication = sections.get("indication", ""),
        comparison = sections.get("comparison", ""),
        technique  = sections.get("technique", ""),
        findings   = sections.get("findings", ""),
        impression = sections.get("impression", ""),
        raw        = report,
    )


def reference_text_for_eval(report: str) -> str:
    """Produce the canonical FINDINGS+IMPRESSION reference string we evaluate
    generated reports against. Falls back to the raw report if both sections
    are missing.
    """
    s = parse(report)
    parts = []
    if s.findings:
        parts.append(f"FINDINGS: {s.findings}")
    if s.impression:
        parts.append(f"IMPRESSION: {s.impression}")
    return "\n".join(parts) if parts else (report or "").strip()
