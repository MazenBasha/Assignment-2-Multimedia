"""Unit tests for the MIMIC-CXR section parser. Pure-Python, no GPU."""

from __future__ import annotations

from data_prep.report_sections import parse, reference_text_for_eval


SAMPLE = """
INDICATION:  Cough and fever.

COMPARISON:  None.

TECHNIQUE: Frontal and lateral views of the chest.

FINDINGS:
The lungs are clear without focal consolidation. No pleural effusion or
pneumothorax. The cardiomediastinal silhouette is normal in size.

IMPRESSION:
1. No acute cardiopulmonary process.
"""


def test_parse_extracts_all_sections() -> None:
    s = parse(SAMPLE)
    assert "Cough and fever" in s.indication
    assert "None" in s.comparison
    assert "Frontal" in s.technique
    assert "No pleural effusion" in s.findings
    assert "No acute" in s.impression
    assert s.has_clinical_content()


def test_parse_unstructured_falls_back_to_findings() -> None:
    s = parse("Some free-text report without any headers.")
    assert s.findings.startswith("Some free-text")
    assert s.impression == ""


def test_reference_for_eval_joins_sections() -> None:
    ref = reference_text_for_eval(SAMPLE)
    assert "FINDINGS:" in ref and "IMPRESSION:" in ref


def test_parse_empty() -> None:
    s = parse("")
    assert not s.has_clinical_content()
