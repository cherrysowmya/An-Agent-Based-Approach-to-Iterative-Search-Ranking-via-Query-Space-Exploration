from pathlib import Path

from search_ranker.data import Document
from search_ranker.ollama_reranker import LLMScore
from search_ranker.rerank_experiment import run_llm_reranking_experiment


class FakeScorer:
    def score(self, *, query: str, document: Document, bm25_score: float) -> LLMScore:
        if document.doc_id == "D2":
            return LLMScore(score=9, reason="Semantic match.")
        return LLMScore(score=2, reason="Weak match.")


def test_llm_reranking_writes_outputs_and_changes_order(tmp_path: Path):
    corpus_path = tmp_path / "corpus.csv"
    queries_path = tmp_path / "queries.csv"
    qrels_path = tmp_path / "qrels.csv"
    output_dir = tmp_path / "rerank"
    corpus_path.write_text(
        "doc_id,title,text\n"
        "D1,Lexical match,bm25 search ranking exact words.\n"
        "D2,Semantic match,passage retrieval relevance evaluation.\n",
        encoding="utf-8",
    )
    queries_path.write_text("query_id,query\nQ1,bm25 search ranking\n", encoding="utf-8")
    qrels_path.write_text("query_id,doc_id,relevance\nQ1,D1,1\nQ1,D2,3\n", encoding="utf-8")

    metrics = run_llm_reranking_experiment(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=output_dir,
        bm25_top_k=2,
        final_top_k=2,
        relevance_threshold=2,
        scorer=FakeScorer(),
    )

    rankings = (output_dir / "reranked_rankings.csv").read_text(encoding="utf-8")
    assert (output_dir / "llm_candidate_scores.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "run_log.txt").exists()
    assert "D2" in rankings.splitlines()[1]
    assert metrics["aggregate"]["mrr@2"] == 1.0
