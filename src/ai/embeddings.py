"""Local embeddings through Ollama. Model-aware: each call uses the selected model
(or the one you pass). No Streamlit, no database here."""

import hashlib
import math
import os
from functools import lru_cache

import ollama
from dotenv import load_dotenv

from src.ai import models

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "60"))
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "6000"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))

EMBED_MODEL = models.DEFAULT_EMBED  # kept for older code; prefer models.get_embed_model()


class EmbeddingError(Exception):
    """User-friendly error for every embedding failure."""


def make_hash(text: str, model: str = None) -> str:
    """Fingerprint of (model, text). A change in either marks the stored vector stale."""
    model = model or models.get_embed_model()
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


def _normalize(vector: list) -> list:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


def _embed(texts: list, model: str, prefix: str) -> list:
    cleaned = [(t or "").strip()[:EMBED_MAX_CHARS] for t in texts]
    if not cleaned or any(not t for t in cleaned):
        raise EmbeddingError("Nothing to embed (empty text).")
    prepared = [prefix + t for t in cleaned]
    vectors = []
    try:
        client = ollama.Client(host=OLLAMA_HOST, timeout=EMBED_TIMEOUT)
        for start in range(0, len(prepared), EMBED_BATCH):
            batch = prepared[start:start + EMBED_BATCH]
            response = client.embed(model=model, input=batch, keep_alive=KEEP_ALIVE)
            vectors.extend(response["embeddings"])
    except ollama.ResponseError as exc:
        if getattr(exc, "status_code", None) == 404:
            raise EmbeddingError(
                f"Embedding model '{model}' is not installed. Run: ollama pull {model}"
            ) from exc
        raise EmbeddingError(f"Ollama rejected the embedding request: {exc}") from exc
    except Exception as exc:  # connection refused, timeout, etc.
        raise EmbeddingError(
            f"Could not reach Ollama at {OLLAMA_HOST} for embeddings. Is it running? ({exc})"
        ) from exc
    if len(vectors) != len(prepared):
        raise EmbeddingError("Ollama returned an unexpected number of embeddings.")
    return [_normalize(list(v)) for v in vectors]


def embed_documents(texts: list, model: str = None) -> list:
    """Embed documents or chunks (model's document prefix). Vectors are unit length."""
    model = model or models.get_embed_model()
    _, doc_prefix = models.embed_prefixes(model)
    return _embed(texts, model, doc_prefix)


@lru_cache(maxsize=512)
def _embed_query_cached(model: str, query: str) -> tuple:
    query_prefix, _ = models.embed_prefixes(model)
    return tuple(_embed([query], model, query_prefix)[0])


def embed_query(query: str, model: str = None) -> list:
    """Embed a search query or concept (model's query prefix). Cached per (model, query)."""
    query = (query or "").strip()
    if not query:
        raise EmbeddingError("Empty search query.")
    return list(_embed_query_cached(model or models.get_embed_model(), query))


def check_embedding_model(model: str = None) -> tuple:
    """(ok, message) for one tiny embedding call."""
    model = model or models.get_embed_model()
    try:
        vector = embed_documents(["ping"], model=model)[0]
    except EmbeddingError as exc:
        return False, str(exc)
    return True, f"Embedding model ready: {model} ({len(vector)} dimensions)"