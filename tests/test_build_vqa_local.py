"""Rule-based VQA extractor smoke tests."""

from data_prep.build_vqa_dataset_local import _pairs_for_report
from data_prep.report_sections import parse


REPORT = """
FINDINGS:
There is a small right pleural effusion with adjacent atelectasis at the
right lung base. No pneumothorax. The cardiomediastinal silhouette is
mildly enlarged.

IMPRESSION:
1. Small right pleural effusion.
2. Mild cardiomegaly.
"""


def test_extracts_presence_yes_for_effusion() -> None:
    sections = parse(REPORT)
    pairs = _pairs_for_report("s001", sections, n_max=20)
    effusion = [p for p in pairs if "pleural effusion" in p["question"].lower()
                and p["question_type"] == "presence"]
    assert len(effusion) == 1
    assert effusion[0]["answer"].lower().startswith("yes")


def test_extracts_presence_no_for_pneumothorax() -> None:
    sections = parse(REPORT)
    pairs = _pairs_for_report("s001", sections, n_max=20)
    pneumo = [p for p in pairs if "pneumothorax" in p["question"].lower()
              and p["question_type"] == "presence"]
    assert len(pneumo) == 1
    assert pneumo[0]["answer"].lower().startswith("no")


def test_extracts_severity_and_side() -> None:
    sections = parse(REPORT)
    pairs = _pairs_for_report("s001", sections, n_max=20)
    types = {p["question_type"] for p in pairs}
    assert "severity"    in types
    assert "location"    in types
    assert "description" in types


def test_empty_report_emits_nothing() -> None:
    pairs = _pairs_for_report("s002", parse(""), n_max=20)
    assert pairs == []
