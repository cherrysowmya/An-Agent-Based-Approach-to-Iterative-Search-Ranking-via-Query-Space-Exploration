"""BM25 plus LLM reranking experiment runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

from search_ranker.bm25 import BM25Retriever
from search_ranker.data import Document, Query, Qrels, load_corpus, load_qrels, load_queries
from search_ranker.metrics import evaluate_rankings
from search_ranker.ollama_reranker import OllamaReranker, RelevanceScorer


def run_llm_reranking_experiment(
    *,
    corpus_path: Union[str, Path],
    queries_path: Union[str, Path],
    qrels_path: Union[str, Path],
    output_dir: Union[str, Path],
    bm25_top_k: int = 20,
    final_top_k: int = 10,
    relevance_threshold: float = 2.0,
    scorer: Optional[RelevanceScorer] = None,
    model: str = "llama3.1:8b",
    ollama_url: str = "http://localhost:11434",
    cache_path: Optional[Union[str, Path]] = None,
    max_document_chars: int = 1800,
    max_queries: Optional[int] = None,
    show_progress: bool = True,
    hybrid_weight: float = 0.0,
) -> Dict[str, object]:
    """Retrieve BM25 candidates, rerank them with an LLM, and write outputs."""
    if bm25_top_k <= 0:
        raise ValueError("bm25_top_k must be greater than 0")
    if final_top_k <= 0:
        raise ValueError("final_top_k must be greater than 0")
    if final_top_k > bm25_top_k:
        raise ValueError("final_top_k cannot be greater than bm25_top_k")
    if not 0.0 <= hybrid_weight <= 1.0:
        raise ValueError("hybrid_weight must be between 0 and 1")

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("max_queries must be greater than 0")
        queries = queries[:max_queries]
    qrels = load_qrels(qrels_path)
    doc_lookup = {document.doc_id: document for document in corpus}
    retriever = BM25Retriever(corpus)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if scorer is None:
        scorer = OllamaReranker(
            model=model,
            base_url=ollama_url,
            cache_path=cache_path or output_path / "llm_score_cache.json",
            max_document_chars=max_document_chars,
        )

    rankings_by_query: Dict[str, List[str]] = {}
    reranked_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []

    for query_index, query in enumerate(queries, start=1):
        if show_progress:
            print(
                f"[{query_index}/{len(queries)}] BM25 candidates for {query.query_id}: "
                f"{query.query}",
                file=sys.stderr,
                flush=True,
            )
        bm25_results = retriever.search(query.query, top_k=bm25_top_k)
        scored_candidates = []
        for bm25_rank, result in enumerate(bm25_results, start=1):
            if show_progress:
                print(
                    f"  scoring candidate {bm25_rank}/{len(bm25_results)}: {result.doc_id}",
                    file=sys.stderr,
                    flush=True,
                )
            document = doc_lookup[result.doc_id]
            llm_score = scorer.score(
                query=query.query,
                document=document,
                bm25_score=result.score,
            )
            scored_candidates.append(
                {
                    "query": query,
                    "document": document,
                    "bm25_rank": bm25_rank,
                    "bm25_score": result.score,
                    "llm_score": llm_score.score,
                    "hybrid_score": _hybrid_score(
                        llm_score=llm_score.score,
                        bm25_rank=bm25_rank,
                        bm25_top_k=len(bm25_results),
                        hybrid_weight=hybrid_weight,
                    ),
                    "llm_reason": llm_score.reason,
                }
            )

        scored_candidates.sort(
            key=lambda row: (-float(row["hybrid_score"]), int(row["bm25_rank"]))
        )
        for rerank_rank, row in enumerate(scored_candidates, start=1):
            document = row["document"]
            candidate_rows.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "rerank_rank": rerank_rank,
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "bm25_rank": row["bm25_rank"],
                    "bm25_score": round(float(row["bm25_score"]), 6),
                    "llm_score": round(float(row["llm_score"]), 6),
                    "hybrid_score": round(float(row["hybrid_score"]), 6),
                    "llm_reason": row["llm_reason"],
                    "relevance": qrels.get(query.query_id, {}).get(document.doc_id, 0.0),
                }
            )
        final_candidates = scored_candidates[:final_top_k]
        rankings_by_query[query.query_id] = [
            str(row["document"].doc_id) for row in final_candidates
        ]
        relevant_docs = qrels.get(query.query_id, {})
        for rerank_rank, row in enumerate(final_candidates, start=1):
            document = row["document"]
            reranked_rows.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "rerank_rank": rerank_rank,
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "bm25_rank": row["bm25_rank"],
                    "bm25_score": round(float(row["bm25_score"]), 6),
                    "llm_score": round(float(row["llm_score"]), 6),
                    "hybrid_score": round(float(row["hybrid_score"]), 6),
                    "llm_reason": row["llm_reason"],
                    "relevance": relevant_docs.get(document.doc_id, 0.0),
                }
            )

    metrics = evaluate_rankings(
        rankings_by_query,
        qrels,
        k=final_top_k,
        relevance_threshold=relevance_threshold,
    )
    _write_reranked_rankings(output_path / "llm_candidate_scores.csv", candidate_rows)
    _write_reranked_rankings(output_path / "reranked_rankings.csv", reranked_rows)
    _write_metrics(output_path / "metrics.json", metrics)
    _write_run_log(
        output_path / "run_log.txt",
        corpus_path=Path(corpus_path),
        queries_path=Path(queries_path),
        qrels_path=Path(qrels_path),
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        relevance_threshold=relevance_threshold,
        model=model,
        ollama_url=ollama_url,
        max_document_chars=max_document_chars,
        max_queries=max_queries,
        hybrid_weight=hybrid_weight,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )
    return metrics


def _write_reranked_rankings(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "query_id",
        "query",
        "rerank_rank",
        "doc_id",
        "title",
        "bm25_rank",
        "bm25_score",
        "llm_score",
        "hybrid_score",
        "llm_reason",
        "relevance",
    ]
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
    bm25_top_k: int,
    final_top_k: int,
    relevance_threshold: float,
    model: str,
    ollama_url: str,
    max_document_chars: int,
    max_queries: Optional[int],
    hybrid_weight: float,
    corpus: List[Document],
    queries: List[Query],
    qrels: Qrels,
) -> None:
    judged_pairs = sum(len(labels) for labels in qrels.values())
    lines = [
        "BM25 + LLM Reranking Run",
        "========================",
        f"corpus_path: {corpus_path}",
        f"queries_path: {queries_path}",
        f"qrels_path: {qrels_path}",
        f"bm25_top_k: {bm25_top_k}",
        f"final_top_k: {final_top_k}",
        f"relevance_threshold: {relevance_threshold}",
        f"model: {model}",
        f"ollama_url: {ollama_url}",
        f"max_document_chars: {max_document_chars}",
        f"max_queries: {max_queries}",
        f"hybrid_weight: {hybrid_weight}",
        "",
        "Dataset Summary",
        "---------------",
        f"documents: {len(corpus)}",
        f"queries: {len(queries)}",
        f"judged query-document pairs: {judged_pairs}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hybrid_score(
    *,
    llm_score: float,
    bm25_rank: int,
    bm25_top_k: int,
    hybrid_weight: float,
) -> float:
    if hybrid_weight == 0:
        return llm_score
    bm25_rank_score = 10.0 * (bm25_top_k - bm25_rank + 1) / bm25_top_k
    return (1 - hybrid_weight) * llm_score + hybrid_weight * bm25_rank_score
