from search_ranker.bm25 import BM25Retriever
from search_ranker.data import Document


def test_bm25_ranks_matching_document_first():
    documents = [
        Document("D1", "Search ranking", "BM25 ranking retrieves documents."),
        Document("D2", "Cooking", "Recipes use vegetables and spices."),
    ]
    retriever = BM25Retriever(documents)

    results = retriever.search("bm25 search", top_k=2)

    assert results[0].doc_id == "D1"
    assert results[0].score > results[1].score

