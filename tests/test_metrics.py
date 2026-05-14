from search_ranker.metrics import ndcg_at_k, precision_at_k, reciprocal_rank


def test_precision_at_k_uses_relevance_threshold():
    ranked = ["D1", "D2", "D3"]
    qrels = {"D1": 0, "D2": 2, "D3": 1}

    assert precision_at_k(ranked, qrels, k=3) == 2 / 3


def test_reciprocal_rank_finds_first_relevant_document():
    ranked = ["D1", "D2", "D3"]
    qrels = {"D2": 1}

    assert reciprocal_rank(ranked, qrels, k=3) == 0.5


def test_ndcg_at_k_uses_graded_relevance():
    ranked = ["D2", "D1", "D3"]
    qrels = {"D1": 3, "D2": 2, "D3": 0}

    score = ndcg_at_k(ranked, qrels, k=3)

    assert 0 < score < 1

