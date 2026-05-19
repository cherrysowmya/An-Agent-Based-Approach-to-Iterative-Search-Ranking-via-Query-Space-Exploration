from pathlib import Path
from typing import List

from search_ranker.data import Document
from search_ranker.ollama_reranker import LLMScore
from search_ranker.suite import run_experiment_suite


class FakeSuiteModel:
    def score(self, *, query: str, document: Document, bm25_score: float) -> LLMScore:
        if "semantic" in query.lower() and document.doc_id == "D2":
            return LLMScore(score=9, reason="Refinement helps.")
        if document.doc_id == "D2":
            return LLMScore(score=7, reason="Relevant semantic passage.")
        return LLMScore(score=2, reason="Weak match.")

    def refine(
        self,
        *,
        original_query: str,
        current_query: str,
        top_documents: List[Document],
        feedback: str,
    ) -> str:
        return "semantic passage retrieval relevance"


def test_experiment_suite_writes_four_experiment_outputs(tmp_path: Path):
    corpus_path = tmp_path / "corpus.csv"
    queries_path = tmp_path / "queries.csv"
    qrels_path = tmp_path / "qrels.csv"
    output_dir = tmp_path / "suite"
    corpus_path.write_text(
        "doc_id,title,text\n"
        "D1,Lexical match,bm25 search ranking exact words.\n"
        "D2,Semantic match,semantic passage retrieval relevance evaluation.\n",
        encoding="utf-8",
    )
    queries_path.write_text("query_id,query\nQ1,bm25 search ranking\n", encoding="utf-8")
    qrels_path.write_text("query_id,doc_id,relevance\nQ1,D1,1\nQ1,D2,3\n", encoding="utf-8")
    model = FakeSuiteModel()

    summary = run_experiment_suite(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=output_dir,
        bm25_top_k=2,
        final_top_k=2,
        max_iterations=2,
        score_threshold=8,
        relevance_threshold=2,
        scorer=model,
        refiner=model,
        quiet=True,
    )

    assert (output_dir / "experiment_1_baseline_vs_reranking.csv").exists()
    assert (output_dir / "experiment_2_method_comparison.csv").exists()
    assert (output_dir / "experiment_3_iteration_analysis.csv").exists()
    assert (output_dir / "experiment_4_query_refinement_impact.csv").exists()
    assert (output_dir / "suite_summary.json").exists()
    assert "experiment_1_baseline_vs_reranking" in summary
    assert "experiment_4_query_refinement_impact" in summary
