"""Pure dataset transformations used by reviewed task adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def add_nanobeir_relevance(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add the conventional positive relevance score without mutating source rows."""
    transformed: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if "score" in item and item["score"] not in (None, 1):
            raise ValueError("NanoBEIR qrel contains a non-positive explicit score")
        item["score"] = 1
        transformed.append(item)
    return transformed


def normalize_parallel_rows(
    rows: Iterable[Mapping[str, Any]], source: str, target: str
) -> list[dict[str, str]]:
    """Normalize direction-specific parallel columns to sentence1/sentence2."""
    if source == target:
        raise ValueError("source and target columns must differ")
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if source not in row or target not in row:
            raise ValueError(f"parallel row {index} lacks {source!r} or {target!r}")
        output.append({"sentence1": str(row[source]), "sentence2": str(row[target])})
    return output


def normalize_parallel_direction(
    rows: Iterable[Mapping[str, Any]], source_language: str
) -> list[dict[str, str]]:
    """Orient mixed-direction ``title/language/translation`` parallel rows."""
    if source_language not in {"en", "ne"}:
        raise ValueError("source_language must be 'en' or 'ne'")
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        language = row.get("language")
        if language not in {"en", "ne"}:
            raise ValueError(f"parallel row {index} has unsupported language {language!r}")
        title, translation = str(row["title"]), str(row["translation"])
        if language == source_language:
            output.append({"sentence1": title, "sentence2": translation})
        else:
            output.append({"sentence1": translation, "sentence2": title})
    return output


def explicit_positive_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    query: str,
    positive: str,
    negatives: str,
) -> list[dict[str, Any]]:
    """Create MTEB reranking samples with the positive included exactly once."""
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        hard_negatives = list(row.get(negatives) or [])
        positive_text = str(row[positive])
        candidates = [positive_text, *(str(value) for value in hard_negatives)]
        if candidates.count(positive_text) != 1:
            raise ValueError(f"reranking row {index} repeats its positive among hard negatives")
        output.append(
            {
                "query": str(row[query]),
                "positive": [positive_text],
                "negative": candidates[1:],
            }
        )
    return output
