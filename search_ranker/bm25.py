"""A small, dependency-free BM25 implementation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, List

from search_ranker.data import Document
from search_ranker.text import tokenize


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    title: str
    score: float


class BM25Retriever:
    """Rank documents with Okapi BM25."""

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("BM25Retriever requires at least one document")

        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(f"{doc.title} {doc.text}") for doc in self.documents]
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        document_count = len(self.documents)
        document_frequencies: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            document_frequencies.update(set(tokens))

        return {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Return the top-k documents for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_terms = tokenize(query)
        scored_results = []
        for index, document in enumerate(self.documents):
            score = self._score_document(query_terms, index)
            scored_results.append(
                SearchResult(doc_id=document.doc_id, title=document.title, score=score)
            )

        scored_results.sort(key=lambda result: (-result.score, result.doc_id))
        return scored_results[:top_k]

    def _score_document(self, query_terms: List[str], doc_index: int) -> float:
        score = 0.0
        term_frequency = self.term_frequencies[doc_index]
        document_length = self.doc_lengths[doc_index]

        for term in query_terms:
            if term not in term_frequency:
                continue

            frequency = term_frequency[term]
            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * document_length / self.avg_doc_length
            )
            score += self.idf.get(term, 0.0) * numerator / denominator

        return score

