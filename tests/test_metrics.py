"""Sanity tests for the eval scorers (token F1, exact-match)."""

from eval.common import exact_match, f1_token, tokenize


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Right Pleural Effusion.") == ["right", "pleural", "effusion"]


def test_f1_perfect_overlap() -> None:
    assert f1_token("no effusion", "no effusion") == 1.0


def test_f1_no_overlap() -> None:
    assert f1_token("yes pneumothorax", "no effusion") == 0.0


def test_exact_match_ignores_punctuation_and_case() -> None:
    assert exact_match("Yes.", "yes") == 1.0
    assert exact_match("yes", "no") == 0.0
