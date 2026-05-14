from search_ranker.text import tokenize


def test_tokenize_lowercases_and_removes_punctuation():
    assert tokenize("BM25 + Search, Ranking!") == ["bm25", "search", "ranking"]

