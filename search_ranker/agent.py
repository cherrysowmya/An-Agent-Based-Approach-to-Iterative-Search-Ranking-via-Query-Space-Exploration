"""Command-line entrypoint for agent-based iterative search refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search_ranker.agent_experiment import run_agent_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BM25 + agent-based iterative refinement with Ollama."
    )
    parser.add_argument("--corpus", required=True, help="CSV with doc_id,title,text columns.")
    parser.add_argument("--queries", required=True, help="CSV with query_id,query columns.")
    parser.add_argument("--qrels", required=True, help="CSV with query_id,doc_id,relevance.")
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=20,
        help="Number of BM25 candidates scored per iteration.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=10,
        help="Number of final documents evaluated per query.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum retrieve-score-plan-refine loops per query.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=6.0,
        help="Average LLM score below which the agent refines the query.",
    )
    parser.add_argument(
        "--improvement-epsilon",
        type=float,
        default=0.25,
        help="Minimum average-score improvement needed to continue after refinement.",
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
        "--cache",
        default=None,
        help="Optional JSON cache path for LLM scores. Defaults to the output directory.",
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
        help="Optional query limit for quick local runs.",
    )
    parser.add_argument(
        "--hybrid-weight",
        type=float,
        default=0.0,
        help="Blend BM25 rank signal into LLM scoring. 0 uses only LLM score.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["llm_score", "ndcg", "mrr", "precision"],
        default="llm_score",
        help="How the agent selects the best iteration for final output.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress messages.")
    parser.add_argument(
        "--out",
        default="outputs/msmarco_trec_dl_2019_agent",
        help="Directory where agent outputs are written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_agent_experiment(
        corpus_path=Path(args.corpus),
        queries_path=Path(args.queries),
        qrels_path=Path(args.qrels),
        output_dir=Path(args.out),
        bm25_top_k=args.bm25_top_k,
        final_top_k=args.final_top_k,
        max_iterations=args.max_iterations,
        score_threshold=args.score_threshold,
        improvement_epsilon=args.improvement_epsilon,
        relevance_threshold=args.relevance_threshold,
        model=args.model,
        ollama_url=args.ollama_url,
        cache_path=Path(args.cache) if args.cache else None,
        max_document_chars=args.max_document_chars,
        max_queries=args.max_queries,
        show_progress=not args.quiet,
        hybrid_weight=args.hybrid_weight,
        selection_metric=args.selection_metric,
    )
    print(json.dumps(result["metrics"]["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
