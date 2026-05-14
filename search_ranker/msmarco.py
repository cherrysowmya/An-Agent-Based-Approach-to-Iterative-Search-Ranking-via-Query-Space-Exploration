"""Materialize MS MARCO/TREC-DL data into the project's CSV format."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Optional, Tuple, Union

DEFAULT_DATASET_ID = "msmarco-passage/trec-dl-2019/judged"
TREC_DL_JUDGED_DATASETS = {
    "2019": "msmarco-passage/trec-dl-2019/judged",
    "2020": "msmarco-passage/trec-dl-2020/judged",
}


def materialize_msmarco_trec_dl(
    *,
    output_dir: Union[str, Path],
    dataset_id: str = DEFAULT_DATASET_ID,
    max_queries: Optional[int] = None,
    include_zero_relevance: bool = True,
) -> Dict[str, object]:
    """Convert an ir_datasets MS MARCO split into corpus/query/qrel CSV files.

    The generated corpus is a judged-document pool, not the full 8.8M passage
    collection. This keeps the current in-memory BM25 baseline runnable while
    preserving the official query and qrel structure for the selected split.
    """
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError(
            "The MS MARCO converter requires ir_datasets. Install it with "
            '`python -m pip install ".[datasets]"`.'
        ) from exc

    dataset = ir_datasets.load(dataset_id)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_queries = {query.query_id: query.text for query in dataset.queries_iter()}
    qrels = list(dataset.qrels_iter())
    query_ids = sorted({qrel.query_id for qrel in qrels})
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("max_queries must be greater than 0")
        query_ids = query_ids[:max_queries]
    selected_query_ids = set(query_ids)

    selected_qrels: Dict[Tuple[str, str], int] = {}
    for qrel in qrels:
        if qrel.query_id not in selected_query_ids:
            continue
        if not include_zero_relevance and qrel.relevance <= 0:
            continue
        key = (qrel.query_id, qrel.doc_id)
        selected_qrels[key] = max(selected_qrels.get(key, qrel.relevance), qrel.relevance)

    selected_doc_ids = sorted({doc_id for _, doc_id in selected_qrels})
    documents = _load_documents(dataset, selected_doc_ids)

    corpus_path = output_path / "corpus.csv"
    queries_path = output_path / "queries.csv"
    qrels_path = output_path / "qrels.csv"
    metadata_path = output_path / "metadata.json"

    _write_corpus(corpus_path, documents)
    _write_queries(queries_path, query_ids, all_queries)
    _write_qrels(qrels_path, selected_qrels)

    metadata = {
        "dataset_id": dataset_id,
        "corpus": str(corpus_path),
        "queries": str(queries_path),
        "qrels": str(qrels_path),
        "query_count": len(query_ids),
        "document_count": len(documents),
        "qrel_count": len(selected_qrels),
        "include_zero_relevance": include_zero_relevance,
        "corpus_note": (
            "This corpus contains the judged-document pool for the selected "
            "queries, not the full MS MARCO passage collection."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def materialize_trec_dl_judged_splits(
    *,
    output_root: Union[str, Path],
    years: Iterable[str] = ("2019", "2020"),
    max_queries: Optional[int] = None,
    include_zero_relevance: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Materialize one or more TREC-DL judged splits under a shared data root."""
    output_root_path = Path(output_root)
    results: Dict[str, Dict[str, object]] = {}
    for year in years:
        if year not in TREC_DL_JUDGED_DATASETS:
            valid_years = ", ".join(sorted(TREC_DL_JUDGED_DATASETS))
            raise ValueError(f"Unknown TREC-DL year {year!r}. Valid years: {valid_years}")
        dataset_id = TREC_DL_JUDGED_DATASETS[year]
        output_dir = output_root_path / f"msmarco_trec_dl_{year}_judged"
        results[year] = materialize_msmarco_trec_dl(
            output_dir=output_dir,
            dataset_id=dataset_id,
            max_queries=max_queries,
            include_zero_relevance=include_zero_relevance,
        )
    manifest_path = output_root_path / "msmarco_trec_dl_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def purge_msmarco_ir_datasets_cache(
    *,
    ir_datasets_home: Optional[Union[str, Path]] = None,
    confirm: bool = False,
) -> List[str]:
    """Remove only the MS MARCO top-level cache created by ir_datasets."""
    if not confirm:
        raise ValueError("Cache purge requires confirm=True")

    cache_home = Path(
        ir_datasets_home
        or os.environ.get("IR_DATASETS_HOME")
        or Path.home() / ".ir_datasets"
    ).expanduser()
    candidates = [
        cache_home / "msmarco-passage",
        cache_home / "msmarco_passage",
        cache_home / "msmarco",
    ]

    removed: List[str] = []
    for candidate in candidates:
        if candidate.exists():
            shutil.rmtree(candidate)
            removed.append(str(candidate))
    return removed


def _load_documents(dataset: object, doc_ids: Iterable[str]) -> Dict[str, str]:
    docstore = dataset.docs_store()
    documents: Dict[str, str] = {}
    missing: List[str] = []
    for doc_id in doc_ids:
        doc = docstore.get(doc_id)
        if doc is None:
            missing.append(doc_id)
            continue
        documents[doc_id] = doc.text
    if missing:
        raise RuntimeError(
            f"Could not load {len(missing)} judged document(s), including {missing[:5]}"
        )
    return documents


def _write_corpus(path: Path, documents: Dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_id", "title", "text"])
        writer.writeheader()
        for doc_id, text in sorted(documents.items()):
            writer.writerow(
                {
                    "doc_id": doc_id,
                    "title": f"MS MARCO Passage {doc_id}",
                    "text": text,
                }
            )


def _write_queries(path: Path, query_ids: Iterable[str], queries: Dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "query"])
        writer.writeheader()
        for query_id in query_ids:
            writer.writerow({"query_id": query_id, "query": queries[query_id]})


def _write_qrels(path: Path, qrels: Dict[Tuple[str, str], int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "doc_id", "relevance"])
        writer.writeheader()
        for query_id, doc_id in sorted(qrels):
            writer.writerow(
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "relevance": qrels[(query_id, doc_id)],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MS MARCO/TREC-DL data into baseline CSV files."
    )
    parser.add_argument(
        "--all-trec-dl-judged",
        action="store_true",
        help="Materialize both msmarco-passage/trec-dl-2019/judged and 2020/judged.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2019", "2020"],
        help="TREC-DL judged years to materialize with --all-trec-dl-judged.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ID,
        help="ir_datasets dataset id to materialize.",
    )
    parser.add_argument(
        "--out",
        default="data/msmarco_trec_dl_2019_judged",
        help=(
            "Directory for a single dataset, or root directory when "
            "--all-trec-dl-judged is set."
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit for fast local experiments.",
    )
    parser.add_argument(
        "--positive-only-qrels",
        action="store_true",
        help="Drop qrels with relevance 0 from the generated qrels file.",
    )
    parser.add_argument(
        "--purge-msmarco-cache",
        action="store_true",
        help="After successful export, remove only the MS MARCO ir_datasets cache.",
    )
    parser.add_argument(
        "--confirm-purge-msmarco-cache",
        action="store_true",
        help="Required with --purge-msmarco-cache to actually delete cache files.",
    )
    parser.add_argument(
        "--ir-datasets-home",
        default=None,
        help="Optional ir_datasets cache home. Defaults to IR_DATASETS_HOME or ~/.ir_datasets.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.all_trec_dl_judged:
        metadata = materialize_trec_dl_judged_splits(
            output_root=args.out,
            years=args.years,
            max_queries=args.max_queries,
            include_zero_relevance=not args.positive_only_qrels,
        )
    else:
        metadata = materialize_msmarco_trec_dl(
            output_dir=args.out,
            dataset_id=args.dataset,
            max_queries=args.max_queries,
            include_zero_relevance=not args.positive_only_qrels,
        )

    if args.purge_msmarco_cache:
        removed = purge_msmarco_ir_datasets_cache(
            ir_datasets_home=args.ir_datasets_home,
            confirm=args.confirm_purge_msmarco_cache,
        )
        metadata = {"exports": metadata, "purged_cache_paths": removed}

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
