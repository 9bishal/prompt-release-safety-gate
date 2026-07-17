"""
Semantic result cache: on repeated CI runs where a PR is only lightly
amended, most (prompt, test_case) pairs haven't actually changed in meaning.
This cache embeds a `prompt_text + question` key and, on a near-duplicate
(cosine similarity above threshold), returns the previously-computed result
instead of calling Groq again — saving both tokens and CI time.

Same interface as the RAG project's cache — swappable for Redis later
without touching callers.
"""

import json
import os

# Used for vector math (cosine similarity calculations)
# pyrefly: ignore [missing-import]
import numpy as np

# Converts text into semantic embeddings
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

from src import config

# Keep a single embedding model in memory instead of loading it repeatedly.
_embedder = None


def _get_embedder() -> SentenceTransformer:
    """Load the embedding model only once and reuse it."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measure how semantically similar two embedding vectors are.

    A score closer to 1.0 means the texts have very similar meaning.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class ResultCache:
    """Simple semantic cache backed by a local JSON file."""

    def __init__(self, path: str = config.CACHE_STORE_PATH):
        self.path = path

        # Load any previously cached results when the cache starts up.
        self.entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        """Read cached entries from disk if they already exist."""
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        """Persist the latest cache contents to disk."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)

    def _key_text(self, prompt_text: str, question: str) -> str:
        """
        Combine the prompt and question into a single string before embedding.

        This lets the cache consider both pieces of context together.
        """
        return f"{prompt_text}\n---\n{question}"

    def get(self, prompt_text: str, question: str) -> dict | None:
        """
        Look for a semantically similar request in the cache.

        Instead of requiring an exact text match, we compare embeddings
        and reuse the previous result if it's similar enough.
        """
        embedder = _get_embedder()
        key_vec = embedder.encode(self._key_text(prompt_text, question))

        best_score, best_entry = 0.0, None

        # Compare against every cached embedding and keep the closest match.
        for entry in self.entries:
            score = _cosine_sim(key_vec, np.array(entry["embedding"]))
            if score > best_score:
                best_score, best_entry = score, entry

        # Return the cached result only if it passes our similarity threshold.
        if best_entry and best_score >= config.CACHE_SIMILARITY_THRESHOLD:
            return best_entry["result"]

        # No good semantic match found.
        return None

    def set(self, prompt_text: str, question: str, result: dict) -> None:
        """
        Store a new result along with its embedding so future requests
        with similar meaning can reuse it.
        """
        embedder = _get_embedder()

        # Generate the embedding and make it JSON-serializable.
        key_vec = embedder.encode(
            self._key_text(prompt_text, question)
        ).tolist()

        self.entries.append(
            {
                "embedding": key_vec,
                "result": result,
            }
        )

        # Save immediately so the cache survives future runs.
        self._save()