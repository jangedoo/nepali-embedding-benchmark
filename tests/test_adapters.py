import pytest

from neb.adapters import (
    normalize_hard_negative_retrieval_rows,
    normalize_hard_negative_rows,
    normalize_nanobeir,
    normalize_parallel_direction,
    normalize_row_retrieval,
    normalize_sts_rows,
)


def test_sts_direction_mapping() -> None:
    row = {
        "sentence1": "English one",
        "sentence2": "English two",
        "sentence1_ne": "नेपाली एक",
        "sentence2_ne": "नेपाली दुई",
        "score": 4,
    }
    assert normalize_sts_rows([row], "ne-ne")[0]["sentence1"] == "नेपाली एक"
    assert normalize_sts_rows([row], "en-ne")[0]["sentence2"] == "नेपाली दुई"
    assert normalize_sts_rows([row], "ne-en")[0]["sentence2"] == "English two"


def test_nanobeir_qrels_default_to_positive_without_mutation() -> None:
    qrels = [{"query-id": "q", "corpus-id": "d"}]
    values = normalize_nanobeir(
        [{"_id": "d", "text": "document"}],
        [{"_id": "q", "text": "query"}],
        qrels,
    )
    assert values["relevant_docs"] == {"q": {"d": 1}}
    assert "score" not in qrels[0]


def test_hard_negative_candidates_are_isolated_per_query() -> None:
    values = normalize_hard_negative_rows(
        [
            {"query": "q1", "positive": "p1", "hard_negative_passages": ["n1"]},
            {"query": "q2", "positive": "p2", "hard_negative_passages": ["n2"]},
        ]
    )
    assert set(values["top_ranked"]["q0"]).isdisjoint(values["top_ranked"]["q1"])
    assert values["relevant_docs"]["q0"] == {"q0:positive": 1}


def test_hard_negatives_are_pooled_for_retrieval() -> None:
    values = normalize_hard_negative_retrieval_rows(
        [
            {"query": "q1", "positive": "p1", "hard_negative_passages": ["n1"]},
            {"query": "q2", "positive": "p2", "hard_negative_passages": ["n2", "n3"]},
        ]
    )
    assert values["top_ranked"] is None
    assert {row["text"] for row in values["corpus"]} == {"p1", "n1", "p2", "n2", "n3"}


def test_row_retrieval_merges_normalized_duplicate_queries_and_documents() -> None:
    values = normalize_row_retrieval(
        [
            {"query": " Same query! ", "positive": "p1", "negative": "shared"},
            {"query": "same query", "positive": "p2", "negative": "shared"},
        ],
        positive_column="positive",
        negative_columns=("negative",),
    )
    assert len(values["queries"]) == 1
    assert len(values["corpus"]) == 3
    assert len(values["relevant_docs"]["q0"]) == 2


def test_hard_negative_positive_leakage_is_rejected() -> None:
    with pytest.raises(ValueError, match="repeats"):
        normalize_hard_negative_rows(
            [{"query": "q", "positive": "same", "hard_negative_passages": ["same"]}]
        )


def test_bitext_orientation() -> None:
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
