from search_ranker.ollama_reranker import OllamaReranker


def test_parse_score_accepts_json_response():
    reranker = OllamaReranker()

    score = reranker._parse_score('{"relevance_score": 8, "reason": "Direct answer."}')

    assert score.score == 8
    assert score.reason == "Direct answer."


def test_parse_score_clamps_score():
    reranker = OllamaReranker()

    score = reranker._parse_score('{"relevance_score": 14, "reason": "Too high."}')

    assert score.score == 10


def test_parse_refined_query_accepts_json_response():
    reranker = OllamaReranker()

    query = reranker._parse_refined_query(
        '{"refined_query": "semantic passage retrieval"}',
        fallback="original query",
    )

    assert query == "semantic passage retrieval"
