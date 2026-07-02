import pytest

from neb.adapters import (
    add_nanobeir_relevance,
    explicit_positive_candidates,
    normalize_parallel_direction,
)


def test_nanobeir_qrels_are_copied_and_scored() -> None:
    source = [{"query-id": "q", "corpus-id": "d"}]
    output = add_nanobeir_relevance(source)
    assert output == [{"query-id": "q", "corpus-id": "d", "score": 1}]
    assert "score" not in source[0]


def test_parallel_direction_normalizes_mixed_rows() -> None:
    rows = [
        {"title": "Hello", "language": "en", "translation": "नमस्ते"},
        {"title": "धन्यवाद", "language": "ne", "translation": "Thanks"},
    ]
    assert normalize_parallel_direction(rows, "en") == [
        {"sentence1": "Hello", "sentence2": "नमस्ते"},
        {"sentence1": "Thanks", "sentence2": "धन्यवाद"},
    ]
    assert normalize_parallel_direction(rows, "ne")[0] == {
        "sentence1": "नमस्ते",
        "sentence2": "Hello",
    }


def test_reranking_rejects_positive_leakage() -> None:
    with pytest.raises(ValueError, match="repeats"):
        explicit_positive_candidates(
            [{"q": "query", "p": "same", "n": ["same"]}],
            query="q",
            positive="p",
            negatives="n",
        )
