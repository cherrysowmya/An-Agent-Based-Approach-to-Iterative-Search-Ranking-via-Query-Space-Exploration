"""Command-line entrypoint for BM25 + local LLM reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search_ranker.rerank_experiment import run_llm_reranking_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BM25 retrieval followed by Ollama LLM reranking."
    )
    parser.add_argument("--corpus", required=True, help="CSV with doc_id,title,text columns.")
    parser.add_argument("--queries", required=True, help="CSV with query_id,query columns.")
    parser.add_argument(
        "--qrels",
        required=True,
        help="CSV with query_id,doc_id,relevance columns.",
    )
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
        help="Number of reranked documents evaluated per query.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=2.0,
        help="Minimum qrel relevance counted as relevant for Precision and MRR.",
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Local Ollama model used for relevance scoring.",
    )
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
        help="Blend BM25 rank signal into reranking score. 0 uses only LLM score.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide per-query and per-candidate progress messages.",
    )
    parser.add_argument(
        "--out",
        default="outputs/msmarco_trec_dl_2019_llm_rerank",
        help="Directory where reranked_rankings.csv, metrics.json, and run_log.txt are written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_llm_reranking_experiment(
        corpus_path=Path(args.corpus),
        queries_path=Path(args.queries),
        qrels_path=Path(args.qrels),
        output_dir=Path(args.out),
        bm25_top_k=args.bm25_top_k,
        final_top_k=args.final_top_k,
        relevance_threshold=args.relevance_threshold,
        model=args.model,
        ollama_url=args.ollama_url,
        cache_path=Path(args.cache) if args.cache else None,
        max_document_chars=args.max_document_chars,
        max_queries=args.max_queries,
        show_progress=not args.quiet,
        hybrid_weight=args.hybrid_weight,
    )
    print(json.dumps(metrics["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
