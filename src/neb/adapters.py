"""Small, pure transformations for NEB's non-native dataset shapes."""

from __future__ import annotations

import re
import unicodedata
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


def _query_key(value: str) -> str:
    """Match duplicate queries without changing the text sent to a model."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", normalized).split())


def normalize_row_retrieval(
    rows: Iterable[Mapping[str, Any]],
    *,
    positive_column: str,
    negative_columns: tuple[str, ...] = (),
    negative_list_column: str | None = None,
) -> dict[str, Any]:
    """Pool row-wise positives and negatives into one retrieval collection.

    Exact document duplicates share an id. Normalized duplicate queries also
    share an id and accumulate every associated positive relevance judgment.
    """
    corpus: list[dict[str, str]] = []
    queries: list[dict[str, str]] = []
    relevant_docs: dict[str, dict[str, int]] = {}
    document_ids: dict[str, str] = {}
    query_ids: dict[str, str] = {}

    def document_id(text: str) -> str:
        if text not in document_ids:
            identifier = f"d{len(document_ids)}"
            document_ids[text] = identifier
            corpus.append({"id": identifier, "title": "", "text": text})
        return document_ids[text]

    if negative_columns and negative_list_column:
        raise ValueError("use scalar or list-valued negative columns, not both")
    required = {"query", positive_column, *negative_columns}
    if negative_list_column:
        required.add(negative_list_column)
    for index, row in enumerate(rows):
        if not required <= row.keys():
            missing = ", ".join(sorted(required - row.keys()))
            raise ValueError(f"retrieval row {index} is missing required columns: {missing}")
        query = str(row["query"]).strip()
        positive = str(row[positive_column]).strip()
        if not query or not positive:
            raise ValueError(f"retrieval row {index} has an empty query or positive")

        key = _query_key(query)
        if not key:
            raise ValueError(f"retrieval row {index} has no normalized query text")
        query_id = query_ids.get(key)
        if query_id is None:
            query_id = f"q{len(query_ids)}"
            query_ids[key] = query_id
            queries.append({"id": query_id, "text": query})
            relevant_docs[query_id] = {}
        relevant_docs[query_id][document_id(positive)] = 1

        negatives = [(column, row[column]) for column in negative_columns]
        if negative_list_column:
            negatives.extend(
                (negative_list_column, value) for value in (row[negative_list_column] or [])
            )
        if negative_list_column and not negatives:
            raise ValueError(f"retrieval row {index} has no candidates")
        for column, value in negatives:
            negative = str(value).strip()
            if not negative:
                raise ValueError(f"retrieval row {index} has an empty {column}")
            document_id(negative)

    if not queries:
        raise ValueError("retrieval collection has no rows")
    return {
        "corpus": corpus,
        "queries": queries,
        "relevant_docs": relevant_docs,
        "top_ranked": None,
    }


def normalize_hard_negative_retrieval_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool list-valued hard negatives for full-corpus retrieval."""
    return normalize_row_retrieval(
        rows,
        positive_column="positive",
        negative_list_column="hard_negative_passages",
    )


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
