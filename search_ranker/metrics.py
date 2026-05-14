"""Ranking-quality metrics for baseline retrieval experiments."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping

from search_ranker.data import Qrels


def precision_at_k(
    ranked_doc_ids: Iterable[str],
    relevant_docs: Mapping[str, float],
    *,
    k: int,
    relevance_threshold: float = 1.0,
) -> float:
    top_docs = list(ranked_doc_ids)[:k]
    if k <= 0:
        raise ValueError("k must be greater than 0")
    relevant_count = sum(
        1 for doc_id in top_docs if relevant_docs.get(doc_id, 0.0) >= relevance_threshold
    )
    return relevant_count / k


def reciprocal_rank(
    ranked_doc_ids: Iterable[str],
    relevant_docs: Mapping[str, float],
    *,
    k: int,
    relevance_threshold: float = 1.0,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    for index, doc_id in enumerate(list(ranked_doc_ids)[:k], start=1):
        if relevant_docs.get(doc_id, 0.0) >= relevance_threshold:
            return 1 / index
    return 0.0


def ndcg_at_k(
    ranked_doc_ids: Iterable[str],
    relevant_docs: Mapping[str, float],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    ranked_relevances = [relevant_docs.get(doc_id, 0.0) for doc_id in list(ranked_doc_ids)[:k]]
    ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]

    actual = _dcg(ranked_relevances)
    ideal = _dcg(ideal_relevances)
    if ideal == 0:
        return 0.0
    return actual / ideal


def _dcg(relevances: Iterable[float]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def evaluate_rankings(
    rankings_by_query: Mapping[str, List[str]],
    qrels: Qrels,
    *,
    k: int,
    relevance_threshold: float = 1.0,
) -> Dict[str, object]:
    """Calculate per-query and aggregate ranking metrics."""
    per_query: Dict[str, Dict[str, float]] = {}
    for query_id, ranked_doc_ids in rankings_by_query.items():
        relevant_docs = qrels.get(query_id, {})
        per_query[query_id] = {
            f"precision@{k}": precision_at_k(
                ranked_doc_ids,
                relevant_docs,
                k=k,
                relevance_threshold=relevance_threshold,
            ),
            f"mrr@{k}": reciprocal_rank(
                ranked_doc_ids,
                relevant_docs,
                k=k,
                relevance_threshold=relevance_threshold,
            ),
            f"ndcg@{k}": ndcg_at_k(ranked_doc_ids, relevant_docs, k=k),
        }

    if not per_query:
        raise ValueError("No rankings were provided for evaluation")

    aggregate = {
        metric_name: sum(metrics[metric_name] for metrics in per_query.values())
        / len(per_query)
        for metric_name in next(iter(per_query.values())).keys()
    }
    return {"aggregate": aggregate, "per_query": per_query}

