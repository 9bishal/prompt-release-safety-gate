"""
Selective test execution: as a golden dataset grows to hundreds of cases,
running all of them on every PR is slow and expensive. This module treats
the golden dataset like a retrieval corpus — it hybrid-searches (BM25 +
vector, RRF-fused) for the cases most relevant to *what actually changed*
in the prompt diff, then reranks with a cross-encoder to get a precise
"smoke test" tier.

This reuses the same hybrid-search + rerank pattern from the RAG project,
applied to a different corpus (test cases instead of documents).
"""
import json
import re

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer, CrossEncoder

from src import config

_embedder = None
_reranker_model = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(config.RERANKER_MODEL)
    return _reranker_model


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_cases(path: str = config.GOLDEN_DATASET_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def diff_prompts(old_prompt: str, new_prompt: str) -> str:
    """Returns a compact textual summary of what changed, used as the retrieval
    query. A line-level diff is enough here — we don't need a full patch, just
    a signal of *which words/topics* changed to steer retrieval."""
    old_lines = set(old_prompt.splitlines())
    new_lines = set(new_prompt.splitlines())
    changed = (old_lines - new_lines) | (new_lines - old_lines)
    return "\n".join(changed) if changed else new_prompt


class GoldenCaseRetriever:
    def __init__(self, cases: list[dict] | None = None):
        self.cases = cases if cases is not None else load_cases()
        corpus_texts = [f"{c['category']} {c['question']}" for c in self.cases]
        self.bm25 = BM25Okapi([_tokenize(t) for t in corpus_texts])

        embedder = _get_embedder()
        self.embeddings = embedder.encode(corpus_texts)

    def _bm25_rank(self, query: str, top_k: int) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    def _vector_rank(self, query: str, top_k: int) -> list[int]:
        embedder = _get_embedder()
        query_vec = embedder.encode(query)
        sims = self.embeddings @ query_vec / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        return list(np.argsort(-sims)[:top_k])

    def _rrf_fuse(self, ranked_lists: list[list[int]], k: int = 60) -> list[int]:
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda i: scores[i], reverse=True)

    def select_relevant_cases(self, diff_query: str, always_include_core: bool = True) -> list[dict]:
        """Returns the smoke-test tier: cases most relevant to the prompt diff,
        via hybrid search + cross-encoder rerank. Always includes any case
        tagged 'core' regardless of relevance, since those are non-negotiable
        regression guards (e.g. out-of-scope refusal behavior)."""
        bm25_ids = self._bm25_rank(diff_query, config.BM25_TOP_K)
        vector_ids = self._vector_rank(diff_query, config.VECTOR_TOP_K)
        fused_ids = self._rrf_fuse([bm25_ids, vector_ids])[: config.FUSION_TOP_K]
        candidates = [self.cases[i] for i in fused_ids]

        reranker = _get_reranker()
        pairs = [[diff_query, f"{c['category']}: {c['question']}"] for c in candidates]
        scores = reranker.predict(pairs) if pairs else []
        for c, s in zip(candidates, scores):
            c["relevance_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c.get("relevance_score", 0), reverse=True)
        selected = ranked[: config.RERANK_TOP_K]

        if always_include_core:
            core_ids = {c["id"] for c in selected}
            for c in self.cases:
                if c["category"] == "out_of_scope" and c["id"] not in core_ids:
                    selected.append(c)

        return selected
