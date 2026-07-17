"""Central configuration for the project.

All tunable settings live here so behavior can be adjusted through
environment variables without changing the application code.
"""

import os
from dotenv import load_dotenv

# Load environment variables from the .env file.
load_dotenv()

# ---------------------------------------------------------------------------
# Groq API configuration
# ---------------------------------------------------------------------------

# API key used to authenticate with Groq.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model used for generating responses.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Separate model used as the evaluation/judge model.
# Keeping this configurable makes it easy to experiment with different judges.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")


# ---------------------------------------------------------------------------
# Local embedding & reranking models
# ---------------------------------------------------------------------------

# Embedding model used for semantic search and caching.
# Runs locally, so there are no API costs for embedding generation.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cross-encoder used to rerank retrieved candidates for better relevance.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------

# Number of keyword-based (BM25) search results to retrieve.
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 10))

# Number of semantic vector search results to retrieve.
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", 10))

# Number of combined results kept after fusion.
FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", 8))

# Final number of results passed through the reranker.
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))


# ---------------------------------------------------------------------------
# Semantic cache settings
# ---------------------------------------------------------------------------

# Minimum cosine similarity required before reusing a cached result.
# Higher values reduce false matches but also reduce cache hits.
CACHE_SIMILARITY_THRESHOLD = float(
    os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.97)
)

# Local file used to persist cached embeddings and results.
CACHE_STORE_PATH = "cache_store.json"


# ---------------------------------------------------------------------------
# Evaluation gate thresholds
# ---------------------------------------------------------------------------

# Maximum acceptable increase in response length (%).
MAX_LENGTH_INCREASE_PCT = float(
    os.getenv("MAX_LENGTH_INCREASE_PCT", 50)
)

# Maximum acceptable increase in estimated API cost (%).
MAX_COST_INCREASE_PCT = float(
    os.getenv("MAX_COST_INCREASE_PCT", 30)
)

# Maximum acceptable increase in latency (%).
MAX_LATENCY_INCREASE_PCT = float(
    os.getenv("MAX_LATENCY_INCREASE_PCT", 50)
)

# Minimum quality score required from the judge model.
MIN_JUDGE_SCORE = float(os.getenv("MIN_JUDGE_SCORE", 0.6))

# Statistical significance threshold for regression testing.
SIGNIFICANCE_ALPHA = float(os.getenv("SIGNIFICANCE_ALPHA", 0.05))


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Location of the golden evaluation dataset.
GOLDEN_DATASET_PATH = "golden_dataset/cases.json"


# ---------------------------------------------------------------------------
# Groq pricing (USD per 1M tokens)
# ---------------------------------------------------------------------------

# Used to estimate inference cost during evaluation.
# These values can be updated if Groq changes its pricing.
GROQ_PRICE_PER_1M_INPUT = 0.05
GROQ_PRICE_PER_1M_OUTPUT = 0.08