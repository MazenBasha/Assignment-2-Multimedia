"""Tests for the JSON-array parser used to filter VQA-generator outputs."""

from data_prep.llm_judge import parse_strict_json_array


def test_plain_array() -> None:
    out = parse_strict_json_array('[{"question": "q", "answer": "a"}]')
    assert len(out) == 1 and out[0]["question"] == "q"


def test_fenced_array() -> None:
    text = '```json\n[{"q":1}]\n```'
    assert parse_strict_json_array(text) == [{"q": 1}]


def test_garbage_returns_empty() -> None:
    assert parse_strict_json_array("the model refused to answer") == []


def test_extra_prose_around_array() -> None:
    text = 'Sure! Here is the JSON:\n[{"a": 1}, {"a": 2}]\nLet me know.'
    assert parse_strict_json_array(text) == [{"a": 1}, {"a": 2}]
