"""Compare BM25 against BM25 plus local LLM reranking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from search_ranker.experiment import run_baseline_experiment
from search_ranker.ollama_reranker import RelevanceScorer
from search_ranker.rerank_experiment import run_llm_reranking_experiment


def compare_bm25_vs_llm_rerank(
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    output_dir: Path,
    bm25_top_k: int = 20,
    final_top_k: int = 10,
    relevance_threshold: float = 2.0,
    model: str = "llama3.1:8b",
    ollama_url: str = "http://localhost:11434",
    max_document_chars: int = 1800,
    max_queries: Optional[int] = None,
    quiet: bool = False,
    scorer: Optional[RelevanceScorer] = None,
    hybrid_weight: float = 0.0,
) -> Dict[str, object]:
    """Run both experiments and write aggregate/per-query comparison files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_dir / "bm25"
    rerank_dir = output_dir / "bm25_llm_rerank"

    baseline_metrics = run_baseline_experiment(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=baseline_dir,
        top_k=final_top_k,
        relevance_threshold=relevance_threshold,
        max_queries=max_queries,
    )
    rerank_metrics = run_llm_reranking_experiment(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=rerank_dir,
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        relevance_threshold=relevance_threshold,
        scorer=scorer,
        model=model,
        ollama_url=ollama_url,
        cache_path=rerank_dir / "llm_score_cache.json",
        max_document_chars=max_document_chars,
        max_queries=max_queries,
        show_progress=not quiet,
        hybrid_weight=hybrid_weight,
    )

    comparison = {
        "bm25": baseline_metrics,
        "bm25_llm_rerank": rerank_metrics,
        "aggregate_delta": _metric_delta(
            baseline_metrics["aggregate"],
            rerank_metrics["aggregate"],
        ),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_aggregate_csv(
        output_dir / "aggregate_comparison.csv",
        baseline_metrics["aggregate"],
        rerank_metrics["aggregate"],
    )
    _write_per_query_csv(
        output_dir / "per_query_comparison.csv",
        baseline_metrics["per_query"],
        rerank_metrics["per_query"],
    )
    return comparison


def _metric_delta(
    baseline: Dict[str, float],
    reranked: Dict[str, float],
) -> Dict[str, float]:
    return {
        metric: reranked[metric] - baseline[metric]
        for metric in sorted(baseline)
        if metric in reranked
    }


def _write_aggregate_csv(
    path: Path,
    baseline: Dict[str, float],
    reranked: Dict[str, float],
) -> None:
    rows = []
    for metric in sorted(baseline):
        if metric not in reranked:
            continue
        rows.append(
            {
                "metric": metric,
                "bm25": baseline[metric],
                "bm25_llm_rerank": reranked[metric],
                "delta": reranked[metric] - baseline[metric],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "bm25", "bm25_llm_rerank", "delta"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_per_query_csv(
    path: Path,
    baseline: Dict[str, Dict[str, float]],
    reranked: Dict[str, Dict[str, float]],
) -> None:
    rows: List[Dict[str, object]] = []
    for query_id in sorted(baseline):
        if query_id not in reranked:
            continue
        for metric in sorted(baseline[query_id]):
            if metric not in reranked[query_id]:
                continue
            rows.append(
                {
                    "query_id": query_id,
                    "metric": metric,
                    "bm25": baseline[query_id][metric],
                    "bm25_llm_rerank": reranked[query_id][metric],
                    "delta": reranked[query_id][metric] - baseline[query_id][metric],
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "metric", "bm25", "bm25_llm_rerank", "delta"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and compare BM25 against BM25 + Ollama LLM reranking."
    )
    parser.add_argument("--corpus", required=True, help="CSV with doc_id,title,text columns.")
    parser.add_argument("--queries", required=True, help="CSV with query_id,query columns.")
    parser.add_argument("--qrels", required=True, help="CSV with query_id,doc_id,relevance.")
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=20,
        help="Number of BM25 candidates sent to the LLM reranker.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=10,
        help="Number of documents evaluated for both systems.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=2.0,
        help="Minimum qrel relevance counted as relevant for Precision and MRR.",
    )
    parser.add_argument("--model", default="llama3.1:8b", help="Local Ollama model.")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--max-document-chars",
        type=int,
        default=1800,
        help="Maximum passage characters sent to the LLM for each candidate.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional query limit for quick local comparisons.",
    )
    parser.add_argument(
        "--hybrid-weight",
        type=float,
        default=0.0,
        help="Blend BM25 rank signal into LLM reranking score.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide LLM reranking progress messages.",
    )
    parser.add_argument(
        "--out",
        default="outputs/msmarco_trec_dl_2019_comparison",
        help="Directory where comparison outputs are written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = compare_bm25_vs_llm_rerank(
        corpus_path=Path(args.corpus),
        queries_path=Path(args.queries),
        qrels_path=Path(args.qrels),
        output_dir=Path(args.out),
        bm25_top_k=args.bm25_top_k,
        final_top_k=args.final_top_k,
        relevance_threshold=args.relevance_threshold,
        model=args.model,
        ollama_url=args.ollama_url,
        max_document_chars=args.max_document_chars,
        max_queries=args.max_queries,
        quiet=args.quiet,
        hybrid_weight=args.hybrid_weight,
    )
    print(json.dumps(comparison["aggregate_delta"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
