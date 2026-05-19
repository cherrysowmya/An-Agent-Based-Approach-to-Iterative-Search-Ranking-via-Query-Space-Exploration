"""Agent-based iterative BM25 retrieval and local LLM refinement."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Optional, Union

from search_ranker.bm25 import BM25Retriever
from search_ranker.data import Document, Query, Qrels, load_corpus, load_qrels, load_queries
from search_ranker.metrics import evaluate_rankings
from search_ranker.ollama_reranker import LLMScore, OllamaReranker, QueryRefiner, RelevanceScorer


@dataclass
class AgentCandidate:
    query_id: str
    iteration: int
    query: str
    doc_id: str
    title: str
    bm25_rank: int
    bm25_score: float
    llm_score: float
    hybrid_score: float
    llm_reason: str
    relevance: float


def run_agent_experiment(
    *,
    corpus_path: Union[str, Path],
    queries_path: Union[str, Path],
    qrels_path: Union[str, Path],
    output_dir: Union[str, Path],
    bm25_top_k: int = 20,
    final_top_k: int = 10,
    max_iterations: int = 3,
    score_threshold: float = 6.0,
    improvement_epsilon: float = 0.25,
    relevance_threshold: float = 2.0,
    scorer: Optional[RelevanceScorer] = None,
    refiner: Optional[QueryRefiner] = None,
    model: str = "llama3.1:8b",
    ollama_url: str = "http://localhost:11434",
    cache_path: Optional[Union[str, Path]] = None,
    max_document_chars: int = 1800,
    max_queries: Optional[int] = None,
    show_progress: bool = True,
    hybrid_weight: float = 0.0,
    selection_metric: str = "llm_score",
) -> Dict[str, object]:
    """Run iterative retrieve-score-plan-refine loops and evaluate final rankings."""
    if bm25_top_k <= 0:
        raise ValueError("bm25_top_k must be greater than 0")
    if final_top_k <= 0:
        raise ValueError("final_top_k must be greater than 0")
    if final_top_k > bm25_top_k:
        raise ValueError("final_top_k cannot be greater than bm25_top_k")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than 0")
    if not 0.0 <= hybrid_weight <= 1.0:
        raise ValueError("hybrid_weight must be between 0 and 1")
    if selection_metric not in {"llm_score", "ndcg", "mrr", "precision"}:
        raise ValueError("selection_metric must be one of llm_score, ndcg, mrr, precision")

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("max_queries must be greater than 0")
        queries = queries[:max_queries]
    qrels = load_qrels(qrels_path)
    retriever = BM25Retriever(corpus)
    doc_lookup = {document.doc_id: document for document in corpus}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if scorer is None or refiner is None:
        ollama = OllamaReranker(
            model=model,
            base_url=ollama_url,
            cache_path=cache_path or output_path / "agent_score_cache.json",
            max_document_chars=max_document_chars,
        )
        scorer = scorer or ollama
        refiner = refiner or ollama

    final_rankings: Dict[str, List[str]] = {}
    final_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    iteration_rows: List[Dict[str, object]] = []

    for query_index, query in enumerate(queries, start=1):
        current_query = query.query
        best_candidates: List[AgentCandidate] = []
        best_selection_score = -1.0
        previous_avg_score: Optional[float] = None

        for iteration in range(1, max_iterations + 1):
            if show_progress:
                print(
                    f"[{query_index}/{len(queries)}] {query.query_id} iteration "
                    f"{iteration}/{max_iterations}: {current_query}",
                    file=sys.stderr,
                    flush=True,
                )

            candidates = _retrieve_and_score(
                query=query,
                current_query=current_query,
                retriever=retriever,
                doc_lookup=doc_lookup,
                qrels=qrels,
                scorer=scorer,
                bm25_top_k=bm25_top_k,
                iteration=iteration,
                show_progress=show_progress,
                hybrid_weight=hybrid_weight,
            )
            candidate_rows.extend(_candidate_to_row(candidate) for candidate in candidates)
            top_candidates = candidates[:final_top_k]
            avg_score = _average_score(top_candidates)
            score_variance = _score_variance(top_candidates)
            improvement = (
                0.0 if previous_avg_score is None else avg_score - previous_avg_score
            )

            selection_score = _selection_score(
                candidates=top_candidates,
                query_id=query.query_id,
                qrels=qrels,
                final_top_k=final_top_k,
                relevance_threshold=relevance_threshold,
                selection_metric=selection_metric,
            )
            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_candidates = top_candidates

            action = _choose_action(
                avg_score=avg_score,
                improvement=improvement,
                iteration=iteration,
                max_iterations=max_iterations,
                score_threshold=score_threshold,
                improvement_epsilon=improvement_epsilon,
            )
            feedback = (
                f"average relevance score={avg_score:.2f}; "
                f"variance={score_variance:.2f}; improvement={improvement:.2f}; "
                f"action={action}"
            )
            next_query = current_query
            if action == "refine_query":
                top_documents = [doc_lookup[candidate.doc_id] for candidate in top_candidates]
                next_query = refiner.refine(
                    original_query=query.query,
                    current_query=current_query,
                    top_documents=top_documents,
                    feedback=feedback,
                )
                if next_query.strip().lower() == current_query.strip().lower():
                    action = "stop"

            iteration_rows.append(
                {
                    "query_id": query.query_id,
                    "iteration": iteration,
                    "query": current_query,
                    "avg_llm_score": round(avg_score, 6),
                    "score_variance": round(score_variance, 6),
                    "improvement": round(improvement, 6),
                    "selection_metric": selection_metric,
                    "selection_score": round(selection_score, 6),
                    "action": action,
                    "next_query": next_query,
                }
            )
            previous_avg_score = avg_score
            current_query = next_query
            if action == "stop":
                break

        final_rankings[query.query_id] = [candidate.doc_id for candidate in best_candidates]
        for rank, candidate in enumerate(best_candidates, start=1):
            row = _candidate_to_row(candidate)
            row["final_rank"] = rank
            final_rows.append(row)

    metrics = evaluate_rankings(
        final_rankings,
        qrels,
        k=final_top_k,
        relevance_threshold=relevance_threshold,
    )
    _write_rows(output_path / "agent_final_rankings.csv", final_rows)
    _write_rows(output_path / "agent_iteration_log.csv", iteration_rows)
    _write_rows(output_path / "agent_candidate_scores.csv", candidate_rows)
    _write_json(output_path / "metrics.json", metrics)
    _write_run_log(
        output_path / "run_log.txt",
        corpus_path=Path(corpus_path),
        queries_path=Path(queries_path),
        qrels_path=Path(qrels_path),
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        max_iterations=max_iterations,
        score_threshold=score_threshold,
        improvement_epsilon=improvement_epsilon,
        relevance_threshold=relevance_threshold,
        model=model,
        max_queries=max_queries,
        hybrid_weight=hybrid_weight,
        selection_metric=selection_metric,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )
    return {
        "metrics": metrics,
        "final_rankings": final_rankings,
        "iteration_rows": iteration_rows,
        "candidate_rows": candidate_rows,
        "final_rows": final_rows,
    }


def _retrieve_and_score(
    *,
    query: Query,
    current_query: str,
    retriever: BM25Retriever,
    doc_lookup: Dict[str, Document],
    qrels: Qrels,
    scorer: RelevanceScorer,
    bm25_top_k: int,
    iteration: int,
    show_progress: bool,
    hybrid_weight: float,
) -> List[AgentCandidate]:
    bm25_results = retriever.search(current_query, top_k=bm25_top_k)
    candidates: List[AgentCandidate] = []
    for bm25_rank, result in enumerate(bm25_results, start=1):
        if show_progress:
            print(
                f"  scoring candidate {bm25_rank}/{len(bm25_results)}: {result.doc_id}",
                file=sys.stderr,
                flush=True,
            )
        document = doc_lookup[result.doc_id]
        llm_score: LLMScore = scorer.score(
            query=current_query,
            document=document,
            bm25_score=result.score,
        )
        candidates.append(
            AgentCandidate(
                query_id=query.query_id,
                iteration=iteration,
                query=current_query,
                doc_id=result.doc_id,
                title=result.title,
                bm25_rank=bm25_rank,
                bm25_score=result.score,
                llm_score=llm_score.score,
                hybrid_score=_hybrid_score(
                    llm_score=llm_score.score,
                    bm25_rank=bm25_rank,
                    bm25_top_k=len(bm25_results),
                    hybrid_weight=hybrid_weight,
                ),
                llm_reason=llm_score.reason,
                relevance=qrels.get(query.query_id, {}).get(result.doc_id, 0.0),
            )
        )
    candidates.sort(key=lambda candidate: (-candidate.hybrid_score, candidate.bm25_rank))
    return candidates


def _choose_action(
    *,
    avg_score: float,
    improvement: float,
    iteration: int,
    max_iterations: int,
    score_threshold: float,
    improvement_epsilon: float,
) -> str:
    if iteration >= max_iterations:
        return "stop"
    if avg_score < score_threshold:
        return "refine_query"
    if iteration > 1 and improvement < improvement_epsilon:
        return "stop"
    return "rerank"


def _average_score(candidates: List[AgentCandidate]) -> float:
    if not candidates:
        return 0.0
    return sum(candidate.llm_score for candidate in candidates) / len(candidates)


def _score_variance(candidates: List[AgentCandidate]) -> float:
    if len(candidates) < 2:
        return 0.0
    return statistics.pvariance(candidate.llm_score for candidate in candidates)


def _selection_score(
    *,
    candidates: List[AgentCandidate],
    query_id: str,
    qrels: Qrels,
    final_top_k: int,
    relevance_threshold: float,
    selection_metric: str,
) -> float:
    if selection_metric == "llm_score":
        return _average_score(candidates)
    ranked_doc_ids = [candidate.doc_id for candidate in candidates[:final_top_k]]
    metrics = evaluate_rankings(
        {query_id: ranked_doc_ids},
        qrels,
        k=final_top_k,
        relevance_threshold=relevance_threshold,
    )["aggregate"]
    return float(metrics[f"{selection_metric}@{final_top_k}"])


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


def _candidate_to_row(candidate: AgentCandidate) -> Dict[str, object]:
    return {
        "query_id": candidate.query_id,
        "iteration": candidate.iteration,
        "query": candidate.query,
        "doc_id": candidate.doc_id,
        "title": candidate.title,
        "bm25_rank": candidate.bm25_rank,
        "bm25_score": round(candidate.bm25_score, 6),
        "llm_score": round(candidate.llm_score, 6),
        "hybrid_score": round(candidate.hybrid_score, 6),
        "llm_reason": candidate.llm_reason,
        "relevance": candidate.relevance,
    }


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run_log(
    path: Path,
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    bm25_top_k: int,
    final_top_k: int,
    max_iterations: int,
    score_threshold: float,
    improvement_epsilon: float,
    relevance_threshold: float,
    model: str,
    max_queries: Optional[int],
    hybrid_weight: float,
    selection_metric: str,
    corpus: List[Document],
    queries: List[Query],
    qrels: Qrels,
) -> None:
    judged_pairs = sum(len(labels) for labels in qrels.values())
    lines = [
        "BM25 + Agent-Based Iterative Refinement Run",
        "===========================================",
        f"corpus_path: {corpus_path}",
        f"queries_path: {queries_path}",
        f"qrels_path: {qrels_path}",
        f"bm25_top_k: {bm25_top_k}",
        f"final_top_k: {final_top_k}",
        f"max_iterations: {max_iterations}",
        f"score_threshold: {score_threshold}",
        f"improvement_epsilon: {improvement_epsilon}",
        f"relevance_threshold: {relevance_threshold}",
        f"model: {model}",
        f"max_queries: {max_queries}",
        f"hybrid_weight: {hybrid_weight}",
        f"selection_metric: {selection_metric}",
        "",
        "Dataset Summary",
        "---------------",
        f"documents: {len(corpus)}",
        f"queries: {len(queries)}",
        f"judged query-document pairs: {judged_pairs}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
