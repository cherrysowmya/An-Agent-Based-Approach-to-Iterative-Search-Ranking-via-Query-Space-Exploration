"""Baseline BM25 experiment runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from search_ranker.bm25 import BM25Retriever
from search_ranker.data import Document, Query, Qrels, load_corpus, load_qrels, load_queries
from search_ranker.metrics import evaluate_rankings


def run_baseline_experiment(
    *,
    corpus_path: Union[str, Path],
    queries_path: Union[str, Path],
    qrels_path: Union[str, Path],
    output_dir: Union[str, Path],
    top_k: int = 5,
    relevance_threshold: float = 1.0,
    max_queries: Optional[int] = None,
) -> Dict[str, object]:
    """Run BM25 retrieval and write rankings, metrics, and run metadata."""
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("max_queries must be greater than 0")
        queries = queries[:max_queries]
    qrels = load_qrels(qrels_path)

    retriever = BM25Retriever(corpus)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rankings_by_query: Dict[str, List[str]] = {}
    ranking_rows: List[Dict[str, object]] = []
    query_lookup = {query.query_id: query.query for query in queries}

    for query in queries:
        results = retriever.search(query.query, top_k=top_k)
        rankings_by_query[query.query_id] = [result.doc_id for result in results]
        relevant_docs = qrels.get(query.query_id, {})
        for rank, result in enumerate(results, start=1):
            ranking_rows.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "rank": rank,
                    "doc_id": result.doc_id,
                    "title": result.title,
                    "score": round(result.score, 6),
                    "relevance": relevant_docs.get(result.doc_id, 0.0),
                }
            )

    metrics = evaluate_rankings(
        rankings_by_query,
        qrels,
        k=top_k,
        relevance_threshold=relevance_threshold,
    )
    _write_rankings(output_path / "rankings.csv", ranking_rows)
    _write_metrics(output_path / "metrics.json", metrics)
    _write_run_log(
        output_path / "run_log.txt",
        corpus_path=Path(corpus_path),
        queries_path=Path(queries_path),
        qrels_path=Path(qrels_path),
        top_k=top_k,
        relevance_threshold=relevance_threshold,
        max_queries=max_queries,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        query_lookup=query_lookup,
    )
    return metrics


def _write_rankings(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["query_id", "query", "rank", "doc_id", "title", "score", "relevance"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics(path: Path, metrics: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_run_log(
    path: Path,
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    top_k: int,
    relevance_threshold: float,
    max_queries: Optional[int],
    corpus: List[Document],
    queries: List[Query],
    qrels: Qrels,
    query_lookup: Dict[str, str],
) -> None:
    judged_pairs = sum(len(labels) for labels in qrels.values())
    lines = [
        "Baseline BM25 Retrieval Run",
        "==========================",
        f"corpus_path: {corpus_path}",
        f"queries_path: {queries_path}",
        f"qrels_path: {qrels_path}",
        f"top_k: {top_k}",
        f"relevance_threshold: {relevance_threshold}",
        f"max_queries: {max_queries}",
        "",
        "Dataset Summary",
        "---------------",
        f"documents: {len(corpus)}",
        f"queries: {len(queries)}",
        f"judged query-document pairs: {judged_pairs}",
        "",
        "Queries",
        "-------",
    ]
    lines.extend(f"{query_id}: {query_text}" for query_id, query_text in query_lookup.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
