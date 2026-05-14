from pathlib import Path

import pytest

from search_ranker.data import load_corpus


def test_load_corpus_validates_required_columns(tmp_path: Path):
    corpus_path = tmp_path / "bad_corpus.csv"
    corpus_path.write_text("doc_id,title\nD1,Missing text\n", encoding="utf-8")

    with pytest.raises(ValueError, match="text"):
        load_corpus(corpus_path)

