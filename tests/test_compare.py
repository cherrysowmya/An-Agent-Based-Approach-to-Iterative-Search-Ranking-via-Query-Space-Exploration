from pathlib import Path

from search_ranker.compare import compare_bm25_vs_llm_rerank
from search_ranker.data import Document
from search_ranker.ollama_reranker import LLMScore


class FakeScorer:
    def score(self, *, query: str, document: Document, bm25_score: float) -> LLMScore:
        if document.doc_id == "D2":
            return LLMScore(score=9, reason="Better semantic match.")
        return LLMScore(score=2, reason="Weaker semantic match.")


def test_compare_writes_aggregate_and_per_query_outputs(tmp_path: Path):
    corpus_path = tmp_path / "corpus.csv"
    queries_path = tmp_path / "queries.csv"
    qrels_path = tmp_path / "qrels.csv"
    output_dir = tmp_path / "comparison"
    corpus_path.write_text(
        "doc_id,title,text\n"
        "D1,Lexical match,bm25 search ranking exact words.\n"
        "D2,Semantic match,passage retrieval relevance evaluation.\n",
        encoding="utf-8",
    )
    queries_path.write_text("query_id,query\nQ1,bm25 search ranking\n", encoding="utf-8")
    qrels_path.write_text("query_id,doc_id,relevance\nQ1,D1,1\nQ1,D2,3\n", encoding="utf-8")

    comparison = compare_bm25_vs_llm_rerank(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=output_dir,
        bm25_top_k=2,
        final_top_k=2,
        relevance_threshold=2,
        max_queries=1,
        quiet=True,
        scorer=FakeScorer(),
    )

    assert (output_dir / "aggregate_comparison.csv").exists()
    assert (output_dir / "per_query_comparison.csv").exists()
    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "bm25" / "metrics.json").exists()
    assert (output_dir / "bm25_llm_rerank" / "metrics.json").exists()
    assert comparison["aggregate_delta"]["mrr@2"] > 0
