"""Small, pure transformations for NEB's non-native dataset shapes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Literal

STSDirection = Literal["ne-ne", "en-ne", "ne-en"]


def normalize_sts_rows(
    rows: Iterable[Mapping[str, Any]], direction: STSDirection
) -> list[dict[str, Any]]:
    """Select and orient one STS-B Nepali language direction."""
    columns = {
        "ne-ne": ("sentence1_ne", "sentence2_ne"),
        "en-ne": ("sentence1", "sentence2_ne"),
        "ne-en": ("sentence1_ne", "sentence2"),
    }
    left, right = columns[direction]
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if left not in row or right not in row or "score" not in row:
            raise ValueError(f"STS row {index} is missing required columns for {direction}")
        output.append(
            {
                "sentence1": str(row[left]),
                "sentence2": str(row[right]),
                "score": float(row["score"]),
            }
        )
    return output


def normalize_nanobeir(
    corpus_rows: Iterable[Mapping[str, Any]],
    query_rows: Iterable[Mapping[str, Any]],
    qrel_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the values required by ``RetrievalSplitData`` for one subset."""
    corpus = []
    for index, row in enumerate(corpus_rows):
        identifier = row.get("id", row.get("_id"))
        if identifier is None or "text" not in row:
            raise ValueError(f"NanoBEIR corpus row {index} lacks an id or text")
        corpus.append(
            {"id": str(identifier), "title": str(row.get("title", "")), "text": str(row["text"])}
        )

    queries = []
    for index, row in enumerate(query_rows):
        identifier = row.get("id", row.get("_id"))
        text = row.get("text", row.get("query"))
        if identifier is None or text is None:
            raise ValueError(f"NanoBEIR query row {index} lacks an id or text")
        queries.append({"id": str(identifier), "text": str(text)})

    relevant_docs: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(qrel_rows):
        query_id = row.get("query-id")
        corpus_id = row.get("corpus-id")
        if query_id is None or corpus_id is None:
            raise ValueError(f"NanoBEIR qrel row {index} lacks query-id or corpus-id")
        score = row.get("score", 1)
        if score is None:
            score = 1
        if int(score) <= 0:
            raise ValueError(f"NanoBEIR qrel row {index} has non-positive relevance")
        relevant_docs[str(query_id)][str(corpus_id)] = int(score)
    return {
        "corpus": corpus,
        "queries": queries,
        "relevant_docs": dict(relevant_docs),
        "top_ranked": None,
    }


def normalize_hard_negative_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create isolated per-query reranking candidate pools.

    Document ids include the query index, preventing candidates from one query
    leaking into another query's ``top_ranked`` pool.
    """
    corpus: list[dict[str, str]] = []
    queries: list[dict[str, str]] = []
    relevant_docs: dict[str, dict[str, int]] = {}
    top_ranked: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        if not {"query", "positive", "hard_negative_passages"} <= row.keys():
            raise ValueError(f"hard-negative row {index} is missing required columns")
        query_id = f"q{index}"
        positive = str(row["positive"])
        negatives = [str(value) for value in (row["hard_negative_passages"] or [])]
        if positive in negatives:
            raise ValueError(f"hard-negative row {index} repeats its positive")
        if not negatives:
            raise ValueError(f"hard-negative row {index} has no candidates")

        positive_id = f"{query_id}:positive"
        candidate_ids = [positive_id]
        corpus.append({"id": positive_id, "title": "", "text": positive})
        for candidate_index, text in enumerate(negatives):
            document_id = f"{query_id}:negative:{candidate_index}"
            candidate_ids.append(document_id)
            corpus.append({"id": document_id, "title": "", "text": text})
        queries.append({"id": query_id, "text": str(row["query"])})
        relevant_docs[query_id] = {positive_id: 1}
        top_ranked[query_id] = candidate_ids
    return {
        "corpus": corpus,
        "queries": queries,
        "relevant_docs": relevant_docs,
        "top_ranked": top_ranked,
    }


def normalize_parallel_direction(
    rows: Iterable[Mapping[str, Any]], source_language: Literal["en", "ne"]
) -> list[dict[str, str]]:
    """Orient mixed ``title/language/translation`` rows for bitext mining."""
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        language = row.get("language")
        if language not in {"en", "ne"}:
            raise ValueError(f"parallel row {index} has unsupported language {language!r}")
        if "title" not in row or "translation" not in row:
            raise ValueError(f"parallel row {index} lacks title or translation")
        title, translation = str(row["title"]), str(row["translation"])
        if language == source_language:
            output.append({"sentence1": title, "sentence2": translation})
        else:
            output.append({"sentence1": translation, "sentence2": title})
    return output
