"""Local Ollama relevance scoring for BM25 candidate reranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, Optional, Protocol, Union
from urllib import error, request

from search_ranker.data import Document


@dataclass(frozen=True)
class LLMScore:
    score: float
    reason: str


class RelevanceScorer(Protocol):
    def score(self, *, query: str, document: Document, bm25_score: float) -> LLMScore:
        """Score one query-document pair."""


class QueryRefiner(Protocol):
    def refine(
        self,
        *,
        original_query: str,
        current_query: str,
        top_documents: list[Document],
        feedback: str,
    ) -> str:
        """Return a rewritten search query."""


class JsonScoreCache:
    """Tiny JSON cache keyed by model/query/document content."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        if self.path.exists():
            self._values: Dict[str, Dict[str, object]] = json.loads(
                self.path.read_text(encoding="utf-8")
            )
        else:
            self._values = {}

    def get(self, key: str) -> Optional[LLMScore]:
        value = self._values.get(key)
        if value is None:
            return None
        return LLMScore(score=float(value["score"]), reason=str(value["reason"]))

    def set(self, key: str, value: LLMScore) -> None:
        self._values[key] = asdict(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._values, indent=2, sort_keys=True) + "\n")


class OllamaReranker:
    """Score relevance with a local Ollama instruction model."""

    def __init__(
        self,
        *,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        max_document_chars: int = 1800,
        cache_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_document_chars = max_document_chars
        self.cache = JsonScoreCache(cache_path) if cache_path is not None else None

    def score(self, *, query: str, document: Document, bm25_score: float) -> LLMScore:
        key = self._cache_key(query=query, document=document)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        prompt = self._build_prompt(query=query, document=document, bm25_score=bm25_score)
        response_text = self._generate(prompt)
        score = self._parse_score(response_text)

        if self.cache is not None:
            self.cache.set(key, score)
        return score

    def _cache_key(self, *, query: str, document: Document) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "query": query,
                "doc_id": document.doc_id,
                "title": document.title,
                "text": document.text,
                "max_document_chars": self.max_document_chars,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_prompt(self, *, query: str, document: Document, bm25_score: float) -> str:
        text = document.text[: self.max_document_chars]
        return (
            "You are reranking search results for an information retrieval experiment.\n"
            "Score how relevant the passage is to the query.\n"
            "Use this scale: 0 = not relevant, 10 = perfectly relevant.\n"
            "Respond with JSON only, with keys relevance_score and reason.\n"
            "The reason must be 25 words or fewer.\n\n"
            f"Query: {query}\n"
            f"BM25 score: {bm25_score:.6f}\n"
            f"Document title: {document.title}\n"
            f"Passage: {text}\n"
        )

    def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama. Start it with `ollama serve` and make sure "
                f"`ollama pull {self.model}` has completed."
            ) from exc

        payload = json.loads(body)
        return str(payload.get("response", ""))

    def _parse_score(self, response_text: str) -> LLMScore:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
            if match is None:
                raise ValueError(f"Ollama response was not JSON: {response_text!r}")
            parsed = json.loads(match.group(0))

        raw_score = parsed.get("relevance_score", parsed.get("score"))
        if raw_score is None:
            raise ValueError(f"Ollama JSON response missing relevance_score: {parsed!r}")
        score = max(0.0, min(10.0, float(raw_score)))
        reason = str(parsed.get("reason", "")).replace("\n", " ").strip()
        return LLMScore(score=score, reason=reason)

    def refine(
        self,
        *,
        original_query: str,
        current_query: str,
        top_documents: list[Document],
        feedback: str,
    ) -> str:
        prompt = self._build_refinement_prompt(
            original_query=original_query,
            current_query=current_query,
            top_documents=top_documents,
            feedback=feedback,
        )
        response_text = self._generate(prompt)
        return self._parse_refined_query(response_text, fallback=current_query)

    def _build_refinement_prompt(
        self,
        *,
        original_query: str,
        current_query: str,
        top_documents: list[Document],
        feedback: str,
    ) -> str:
        snippets = []
        for index, document in enumerate(top_documents[:5], start=1):
            snippets.append(
                f"{index}. {document.title}: {document.text[:350]}"
            )
        return (
            "You are improving a search query for a BM25 retrieval experiment.\n"
            "Rewrite the query to improve retrieval while preserving the original intent.\n"
            "Use concise keyword-rich search terms. Do not answer the query.\n"
            "Respond with JSON only, with key refined_query.\n\n"
            f"Original query: {original_query}\n"
            f"Current query: {current_query}\n"
            f"Feedback: {feedback}\n"
            "Top retrieved passages:\n"
            + "\n".join(snippets)
        )

    def _parse_refined_query(self, response_text: str, *, fallback: str) -> str:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
            if match is None:
                return fallback
            parsed = json.loads(match.group(0))

        refined_query = str(parsed.get("refined_query", "")).strip()
        return refined_query or fallback
