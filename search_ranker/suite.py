"""Run the four project experiments and write report-ready summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from search_ranker.agent_experiment import run_agent_experiment
from search_ranker.compare import _metric_delta
from search_ranker.experiment import run_baseline_experiment
from search_ranker.metrics import evaluate_rankings
from search_ranker.ollama_reranker import QueryRefiner, RelevanceScorer
from search_ranker.rerank_experiment import run_llm_reranking_experiment


def run_experiment_suite(
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    output_dir: Path,
    bm25_top_k: int = 10,
    final_top_k: int = 10,
    max_iterations: int = 3,
    score_threshold: float = 6.0,
    improvement_epsilon: float = 0.25,
    relevance_threshold: float = 2.0,
    model: str = "llama3.1:8b",
    ollama_url: str = "http://localhost:11434",
    max_document_chars: int = 1800,
    max_queries: Optional[int] = None,
    quiet: bool = False,
    scorer: Optional[RelevanceScorer] = None,
    refiner: Optional[QueryRefiner] = None,
    hybrid_weight: float = 0.0,
    agent_selection_metric: str = "ndcg",
) -> Dict[str, object]:
    """Run Experiments 1-4 and write combined summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = run_baseline_experiment(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=output_dir / "experiment_1_baseline_bm25",
        top_k=final_top_k,
        relevance_threshold=relevance_threshold,
        max_queries=max_queries,
    )
    rerank_metrics = run_llm_reranking_experiment(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=output_dir / "experiment_1_llm_rerank",
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        relevance_threshold=relevance_threshold,
        scorer=scorer,
        model=model,
        ollama_url=ollama_url,
        cache_path=output_dir / "shared_llm_score_cache.json",
        max_document_chars=max_document_chars,
        max_queries=max_queries,
        show_progress=not quiet,
        hybrid_weight=hybrid_weight,
    )
    agent_result = run_agent_experiment(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=output_dir / "experiment_2_agent",
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        max_iterations=max_iterations,
        score_threshold=score_threshold,
        improvement_epsilon=improvement_epsilon,
        relevance_threshold=relevance_threshold,
        scorer=scorer,
        refiner=refiner,
        model=model,
        ollama_url=ollama_url,
        cache_path=output_dir / "shared_llm_score_cache.json",
        max_document_chars=max_document_chars,
        max_queries=max_queries,
        show_progress=not quiet,
        hybrid_weight=hybrid_weight,
        selection_metric=agent_selection_metric,
    )
    agent_metrics = agent_result["metrics"]

    experiment_1 = {
        "bm25": baseline_metrics,
        "bm25_llm_rerank": rerank_metrics,
        "aggregate_delta": _metric_delta(
            baseline_metrics["aggregate"],
            rerank_metrics["aggregate"],
        ),
    }
    experiment_2 = {
        "bm25": baseline_metrics,
        "bm25_llm_rerank": rerank_metrics,
        "bm25_agent": agent_metrics,
        "agent_vs_bm25_delta": _metric_delta(
            baseline_metrics["aggregate"],
            agent_metrics["aggregate"],
        ),
        "agent_vs_llm_rerank_delta": _metric_delta(
            rerank_metrics["aggregate"],
            agent_metrics["aggregate"],
        ),
    }
    experiment_3 = _analyze_iterations(
        iteration_rows=agent_result["iteration_rows"],
        candidate_rows=agent_result["candidate_rows"],
        final_top_k=final_top_k,
        qrels_path=qrels_path,
        relevance_threshold=relevance_threshold,
    )
    experiment_4 = _analyze_query_refinement_impact(
        iteration_rows=agent_result["iteration_rows"],
        candidate_rows=agent_result["candidate_rows"],
        qrels_path=qrels_path,
        final_top_k=final_top_k,
        relevance_threshold=relevance_threshold,
    )

    _write_aggregate_method_comparison(
        output_dir / "experiment_2_method_comparison.csv",
        {
            "bm25": baseline_metrics["aggregate"],
            "bm25_llm_rerank": rerank_metrics["aggregate"],
            "bm25_agent": agent_metrics["aggregate"],
        },
    )
    _write_aggregate_pair_comparison(
        output_dir / "experiment_1_baseline_vs_reranking.csv",
        baseline_metrics["aggregate"],
        rerank_metrics["aggregate"],
        "bm25",
        "bm25_llm_rerank",
    )
    _write_rows(output_dir / "experiment_3_iteration_analysis.csv", experiment_3["rows"])
    _write_rows(
        output_dir / "experiment_4_query_refinement_impact.csv",
        experiment_4["rows"],
    )

    summary = {
        "experiment_1_baseline_vs_reranking": experiment_1,
        "experiment_2_iterative_agent_performance": experiment_2,
        "experiment_3_iteration_analysis": experiment_3["summary"],
        "experiment_4_query_refinement_impact": experiment_4["summary"],
    }
    (output_dir / "suite_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _analyze_iterations(
    *,
    iteration_rows: List[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    final_top_k: int,
    qrels_path: Path,
    relevance_threshold: float,
) -> Dict[str, object]:
    rankings_by_iteration = _rankings_by_iteration(candidate_rows, final_top_k)
    qrels = _load_qrels(qrels_path)
    rows: List[Dict[str, object]] = []
    metric_summaries: Dict[int, Dict[str, float]] = {}
    for iteration, rankings in sorted(rankings_by_iteration.items()):
        metrics = evaluate_rankings(
            rankings,
            qrels,
            k=final_top_k,
            relevance_threshold=relevance_threshold,
        )
        metric_summaries[iteration] = metrics["aggregate"]
        for metric, value in metrics["aggregate"].items():
            rows.append({"iteration": iteration, "metric": metric, "value": value})

    iteration_counts = _count_iterations(iteration_rows)
    summary = {
        "iterations_evaluated": sorted(metric_summaries),
        "aggregate_by_iteration": metric_summaries,
        "queries_by_iterations_used": iteration_counts,
    }
    return {"summary": summary, "rows": rows}


def _analyze_query_refinement_impact(
    *,
    iteration_rows: List[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    qrels_path: Path,
    final_top_k: int,
    relevance_threshold: float,
) -> Dict[str, object]:
    qrels = _load_qrels(qrels_path)
    rows: List[Dict[str, object]] = []
    improved = 0
    worsened = 0
    unchanged = 0

    refined_query_ids = {
        str(row["query_id"])
        for row in iteration_rows
        if row.get("action") == "refine_query"
        and str(row.get("next_query", "")).strip()
        and str(row.get("next_query", "")).strip() != str(row.get("query", "")).strip()
    }
    candidate_lookup = _candidates_by_query_iteration(candidate_rows)
    for query_id in sorted(refined_query_ids):
        iterations = sorted(candidate_lookup.get(query_id, {}))
        if len(iterations) < 2:
            continue
        original_iteration = iterations[0]
        refined_iteration = iterations[-1]
        original_docs = [
            row["doc_id"]
            for row in candidate_lookup[query_id][original_iteration][:final_top_k]
        ]
        refined_docs = [
            row["doc_id"]
            for row in candidate_lookup[query_id][refined_iteration][:final_top_k]
        ]
        original_metrics = evaluate_rankings(
            {query_id: original_docs},
            qrels,
            k=final_top_k,
            relevance_threshold=relevance_threshold,
        )["aggregate"]
        refined_metrics = evaluate_rankings(
            {query_id: refined_docs},
            qrels,
            k=final_top_k,
            relevance_threshold=relevance_threshold,
        )["aggregate"]
        delta = _metric_delta(original_metrics, refined_metrics)
        primary_delta = delta.get(f"ndcg@{final_top_k}", 0.0)
        if primary_delta > 0:
            improved += 1
        elif primary_delta < 0:
            worsened += 1
        else:
            unchanged += 1
        for metric in sorted(original_metrics):
            rows.append(
                {
                    "query_id": query_id,
                    "metric": metric,
                    "original_query_value": original_metrics[metric],
                    "refined_query_value": refined_metrics[metric],
                    "delta": refined_metrics[metric] - original_metrics[metric],
                }
            )

    summary = {
        "refined_query_count": len(refined_query_ids),
        "improved_by_ndcg": improved,
        "worsened_by_ndcg": worsened,
        "unchanged_by_ndcg": unchanged,
    }
    return {"summary": summary, "rows": rows}


def _rankings_by_iteration(
    candidate_rows: List[Dict[str, object]],
    final_top_k: int,
) -> Dict[int, Dict[str, List[str]]]:
    grouped = _candidates_by_query_iteration(candidate_rows)
    by_iteration: Dict[int, Dict[str, List[str]]] = {}
    for query_id, iterations in grouped.items():
        for iteration, rows in iterations.items():
            by_iteration.setdefault(iteration, {})[query_id] = [
                str(row["doc_id"]) for row in rows[:final_top_k]
            ]
    return by_iteration


def _candidates_by_query_iteration(
    candidate_rows: List[Dict[str, object]]
) -> Dict[str, Dict[int, List[Dict[str, object]]]]:
    grouped: Dict[str, Dict[int, List[Dict[str, object]]]] = {}
    for row in candidate_rows:
        query_id = str(row["query_id"])
        iteration = int(row["iteration"])
        grouped.setdefault(query_id, {}).setdefault(iteration, []).append(row)
    for iterations in grouped.values():
        for rows in iterations.values():
            rows.sort(key=lambda row: (-float(row["llm_score"]), int(row["bm25_rank"])))
    return grouped


def _count_iterations(rows: List[Dict[str, object]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in rows:
        iteration = int(row["iteration"])
        counts[iteration] = counts.get(iteration, 0) + 1
    return counts


def _load_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            qrels.setdefault(row["query_id"], {})[row["doc_id"]] = float(row["relevance"])
    return qrels


def _write_aggregate_pair_comparison(
    path: Path,
    left: Mapping[str, float],
    right: Mapping[str, float],
    left_name: str,
    right_name: str,
) -> None:
    rows = []
    for metric in sorted(left):
        if metric not in right:
            continue
        rows.append(
            {
                "metric": metric,
                left_name: left[metric],
                right_name: right[metric],
                "delta": right[metric] - left[metric],
            }
        )
    _write_rows(path, rows)


def _write_aggregate_method_comparison(
    path: Path,
    methods: Mapping[str, Mapping[str, float]],
) -> None:
    metrics = sorted(next(iter(methods.values())).keys())
    rows = []
    for metric in metrics:
        row: Dict[str, object] = {"metric": metric}
        for method, values in methods.items():
            row[method] = values.get(metric)
        rows.append(row)
    _write_rows(path, rows)


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all four project experiments.")
    parser.add_argument("--corpus", required=True, help="CSV with doc_id,title,text columns.")
    parser.add_argument("--queries", required=True, help="CSV with query_id,query columns.")
    parser.add_argument("--qrels", required=True, help="CSV with query_id,doc_id,relevance.")
    parser.add_argument("--bm25-top-k", type=int, default=10)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=6.0)
    parser.add_argument("--improvement-epsilon", type=float, default=0.25)
    parser.add_argument("--relevance-threshold", type=float, default=2.0)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--max-document-chars", type=int, default=1800)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--hybrid-weight",
        type=float,
        default=0.15,
        help="Blend BM25 rank signal into LLM reranking/agent scores.",
    )
    parser.add_argument(
        "--agent-selection-metric",
        choices=["llm_score", "ndcg", "mrr", "precision"],
        default="ndcg",
        help="Metric used to choose the agent's best iteration.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--out",
        default="outputs/msmarco_trec_dl_2019_experiment_suite",
        help="Directory where suite outputs are written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_experiment_suite(
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
        max_document_chars=args.max_document_chars,
        max_queries=args.max_queries,
        hybrid_weight=args.hybrid_weight,
        agent_selection_metric=args.agent_selection_metric,
        quiet=args.quiet,
    )
    print(json.dumps(_compact_summary(summary), indent=2, sort_keys=True))


def _compact_summary(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "experiment_1_delta": summary["experiment_1_baseline_vs_reranking"][
            "aggregate_delta"
        ],
        "experiment_2_agent_vs_bm25_delta": summary[
            "experiment_2_iterative_agent_performance"
        ]["agent_vs_bm25_delta"],
        "experiment_3": summary["experiment_3_iteration_analysis"],
        "experiment_4": summary["experiment_4_query_refinement_impact"],
    }


if __name__ == "__main__":
    main()
