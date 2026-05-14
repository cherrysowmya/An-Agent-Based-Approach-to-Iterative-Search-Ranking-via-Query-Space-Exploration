"""CSV loaders for corpora, queries, and relevance labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Union


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    query_id: str
    query: str


Qrels = Dict[str, Dict[str, float]]


def _read_csv(path: Path, required_columns: set[str]) -> List[Mapping[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = required_columns - columns
        if missing:
            required = ", ".join(sorted(required_columns))
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"{path} is missing required column(s): {missing_text}. "
                f"Expected columns: {required}"
            )
        return list(reader)


def load_corpus(path: Union[str, Path]) -> List[Document]:
    """Load documents from a CSV with columns doc_id,title,text."""
    rows = _read_csv(Path(path), {"doc_id", "title", "text"})
    documents = [
        Document(
            doc_id=row["doc_id"].strip(),
            title=row["title"].strip(),
            text=row["text"].strip(),
        )
        for row in rows
    ]
    if not documents:
        raise ValueError(f"{path} does not contain any documents")
    return documents


def load_queries(path: Union[str, Path]) -> List[Query]:
    """Load search queries from a CSV with columns query_id,query."""
    rows = _read_csv(Path(path), {"query_id", "query"})
    queries = [
        Query(query_id=row["query_id"].strip(), query=row["query"].strip())
        for row in rows
    ]
    if not queries:
        raise ValueError(f"{path} does not contain any queries")
    return queries


def load_qrels(path: Union[str, Path]) -> Qrels:
    """Load graded relevance labels from query_id,doc_id,relevance CSV rows."""
    rows = _read_csv(Path(path), {"query_id", "doc_id", "relevance"})
    qrels: Qrels = {}
    for row in rows:
        query_id = row["query_id"].strip()
        doc_id = row["doc_id"].strip()
        try:
            relevance = float(row["relevance"])
        except ValueError as exc:
            raise ValueError(
                f"Invalid relevance value for query_id={query_id}, doc_id={doc_id}: "
                f"{row['relevance']!r}"
            ) from exc
        qrels.setdefault(query_id, {})[doc_id] = relevance
    return qrels
