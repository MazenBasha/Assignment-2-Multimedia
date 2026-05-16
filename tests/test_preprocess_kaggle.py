"""Parser smoke test on the exact Kaggle CSV format we observed
(simhadrisadaram/mimic-cxr-dataset).
"""

from data_prep.preprocess_kaggle import _extract_study_id, _safe_literal_eval


def test_extract_study_id_from_canonical_path() -> None:
    p = "files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
    assert _extract_study_id(p) == "s50414267"


def test_extract_study_id_handles_no_study() -> None:
    assert _extract_study_id("files/p10/p10000032/random.jpg") is None


def test_safe_literal_eval_parses_kaggle_string() -> None:
    s = "['files/p10/p10000032/s50414267/a.jpg', 'files/p10/p10000032/s50414267/b.jpg']"
    out = _safe_literal_eval(s)
    assert out == ["files/p10/p10000032/s50414267/a.jpg",
                   "files/p10/p10000032/s50414267/b.jpg"]


def test_safe_literal_eval_handles_garbage() -> None:
    assert _safe_literal_eval("not a list") == []
    assert _safe_literal_eval(None) == []
    assert _safe_literal_eval(42) == []


def test_safe_literal_eval_handles_already_a_list() -> None:
    assert _safe_literal_eval(["a", "b"]) == ["a", "b"]
