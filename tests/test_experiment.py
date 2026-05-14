from pathlib import Path

from search_ranker.experiment import run_baseline_experiment


def test_baseline_experiment_writes_outputs(tmp_path: Path):
    output_dir = tmp_path / "baseline"
    corpus_path = tmp_path / "corpus.csv"
    queries_path = tmp_path / "queries.csv"
    qrels_path = tmp_path / "qrels.csv"
    corpus_path.write_text(
        "doc_id,title,text\n"
        "D1,Search ranking,BM25 retrieves relevant search passages.\n"
        "D2,Cooking,Vegetable soup recipes use fresh ingredients.\n",
        encoding="utf-8",
    )
    queries_path.write_text("query_id,query\nQ1,bm25 search ranking\n", encoding="utf-8")
    qrels_path.write_text("query_id,doc_id,relevance\nQ1,D1,3\nQ1,D2,0\n", encoding="utf-8")

    metrics = run_baseline_experiment(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=output_dir,
        top_k=2,
        relevance_threshold=2,
    )

    assert (output_dir / "rankings.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "run_log.txt").exists()
    assert "aggregate" in metrics
