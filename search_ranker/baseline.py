"""Command-line entrypoint for the BM25 baseline experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search_ranker.experiment import run_baseline_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the BM25 baseline search ranking experiment."
    )
    parser.add_argument("--corpus", required=True, help="CSV with doc_id,title,text columns.")
    parser.add_argument("--queries", required=True, help="CSV with query_id,query columns.")
    parser.add_argument(
        "--qrels",
        required=True,
        help="CSV with query_id,doc_id,relevance columns.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve and evaluate per query.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=1.0,
        help="Minimum qrel relevance counted as relevant for Precision and MRR.",
    )
    parser.add_argument(
        "--out",
        default="outputs/msmarco_trec_dl_2019_baseline",
        help="Directory where rankings.csv, metrics.json, and run_log.txt are written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_baseline_experiment(
        corpus_path=Path(args.corpus),
        queries_path=Path(args.queries),
        qrels_path=Path(args.qrels),
        output_dir=Path(args.out),
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
    )
    print(json.dumps(metrics["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
