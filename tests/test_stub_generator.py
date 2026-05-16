"""Smoke tests for the nearest-neighbour stub generator."""

from PIL import Image

from generation.stub_generator import StubGenerator


def _img():
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


def _rag_prompt_with(reports: list[str]) -> str:
    lines = []
    for i, r in enumerate(reports):
        lines.append(f"[#{i+1}  similarity=0.{i+5}  study_id=s{i}]")
        lines.append(r)
        lines.append("")
    return "REFERENCE REPORTS:\n" + "\n".join(lines)


def test_mode_a_returns_first_retrieved_report() -> None:
    g = StubGenerator()
    user = _rag_prompt_with([
        "FINDINGS: clear lungs.\nIMPRESSION: normal.",
        "FINDINGS: effusion.",
    ])
    out = g.generate(_img(), system="...", user=user)
    assert "clear lungs" in out
    assert "effusion" not in out  # only the FIRST report


def test_mode_b_presence_yes_when_finding_in_top_report() -> None:
    g = StubGenerator()
    user = _rag_prompt_with(["FINDINGS: small right pleural effusion."]) + \
           "\n\nCURRENT X-RAY: <image>\nQUESTION: Is there pleural effusion?\nANSWER:"
    out = g.generate(_img(), system="...", user=user)
    assert out.lower().startswith("yes")


def test_mode_b_presence_no_when_negated() -> None:
    g = StubGenerator()
    user = _rag_prompt_with(["FINDINGS: no pleural effusion."]) + \
           "\n\nCURRENT X-RAY: <image>\nQUESTION: Is there pleural effusion?\nANSWER:"
    out = g.generate(_img(), system="...", user=user)
    assert out.lower().startswith("no")


def test_mode_b_severity() -> None:
    g = StubGenerator()
    user = _rag_prompt_with(["FINDINGS: mild cardiomegaly."]) + \
           "\n\nCURRENT X-RAY: <image>\nQUESTION: How severe is the cardiomegaly?\nANSWER:"
    out = g.generate(_img(), system="...", user=user)
    assert out.lower().startswith("mild")
