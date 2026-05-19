from pathlib import Path
from typing import List

from search_ranker.agent_experiment import run_agent_experiment
from search_ranker.data import Document
from search_ranker.ollama_reranker import LLMScore


class FakeAgentModel:
    def score(self, *, query: str, document: Document, bm25_score: float) -> LLMScore:
        if "semantic" in query.lower() and document.doc_id == "D2":
            return LLMScore(score=9, reason="Refined query finds semantic match.")
        if document.doc_id == "D1":
            return LLMScore(score=4, reason="Lexical but weak.")
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


def test_agent_experiment_refines_and_writes_iteration_logs(tmp_path: Path):
    corpus_path = tmp_path / "corpus.csv"
    queries_path = tmp_path / "queries.csv"
    qrels_path = tmp_path / "qrels.csv"
    output_dir = tmp_path / "agent"
    corpus_path.write_text(
        "doc_id,title,text\n"
        "D1,Lexical match,bm25 search ranking exact words.\n"
        "D2,Semantic match,semantic passage retrieval relevance evaluation.\n",
        encoding="utf-8",
    )
    queries_path.write_text("query_id,query\nQ1,bm25 search ranking\n", encoding="utf-8")
    qrels_path.write_text("query_id,doc_id,relevance\nQ1,D1,1\nQ1,D2,3\n", encoding="utf-8")

    model = FakeAgentModel()
    result = run_agent_experiment(
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        output_dir=output_dir,
        bm25_top_k=2,
        final_top_k=2,
        max_iterations=2,
        score_threshold=6,
        relevance_threshold=2,
        scorer=model,
        refiner=model,
        show_progress=False,
    )

    iteration_log = (output_dir / "agent_iteration_log.csv").read_text(encoding="utf-8")
    final_rankings = (output_dir / "agent_final_rankings.csv").read_text(encoding="utf-8")
    assert (output_dir / "agent_candidate_scores.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert "refine_query" in iteration_log
    assert "D2" in final_rankings.splitlines()[1]
    assert result["metrics"]["aggregate"]["mrr@2"] == 1.0
