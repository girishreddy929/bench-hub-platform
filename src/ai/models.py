"""Model registry + the currently selected models (process-wide). No Streamlit here."""

import os
import threading

import ollama
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_SUMMARY = os.getenv("OLLAMA_MODEL", "phi4-mini")
DEFAULT_EMBED = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

# name -> label shown in the dropdown
SUMMARY_MODELS = {
    "phi4-mini": "Phi-4 mini 3.8B (default, fast)",
    "qwen2.5:3b": "Qwen 2.5 3B (faster, lighter)",
    "qwen2.5:7b": "Qwen 2.5 7B (slower, compare quality)",
    "llama3.1:8b": "Llama 3.1 8B (alternative)",
}

# name -> label + prefixes. Each embedding model expects its own prefix format.
EMBED_MODELS = {
    "qwen3-embedding:0.6b": {
        "label": "Qwen3 Embedding 0.6B (default, best quality)",
        "query_prefix": "Instruct: Given a skill or role query, retrieve relevant employee profiles\nQuery:",
        "doc_prefix": "",
        "skill_floor": 0.25, "skill_ceiling": 0.60,
        "chunk_floor": 0.30, "chunk_ceiling": 0.60,
    },
    "nomic-embed-text": {
        "label": "Nomic Embed Text (768 dims, long text)",
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
        "skill_floor": 0.55, "skill_ceiling": 0.78,
        "chunk_floor": 0.50, "chunk_ceiling": 0.75,
    },
}

# A model named in .env but missing above still works (no prefixes).
SUMMARY_MODELS.setdefault(DEFAULT_SUMMARY, DEFAULT_SUMMARY)
EMBED_MODELS.setdefault(DEFAULT_EMBED, {"label": DEFAULT_EMBED, "query_prefix": "", "doc_prefix": ""})

_lock = threading.Lock()
_selected = {"summary": DEFAULT_SUMMARY, "embed": DEFAULT_EMBED}


def summary_choices() -> list:
    return [DEFAULT_SUMMARY] + [m for m in SUMMARY_MODELS if m != DEFAULT_SUMMARY]


def embed_choices() -> list:
    return [DEFAULT_EMBED] + [m for m in EMBED_MODELS if m != DEFAULT_EMBED]


def summary_label(name: str) -> str:
    return SUMMARY_MODELS.get(name, name)


def embed_label(name: str) -> str:
    return EMBED_MODELS.get(name, {}).get("label", name)


def get_summary_model() -> str:
    with _lock:
        return _selected["summary"]


def get_embed_model() -> str:
    with _lock:
        return _selected["embed"]


def set_summary_model(name: str) -> None:
    if name not in SUMMARY_MODELS:
        raise ValueError(f"Unknown summary model: {name}")
    with _lock:
        _selected["summary"] = name


def set_embed_model(name: str) -> None:
    if name not in EMBED_MODELS:
        raise ValueError(f"Unknown embedding model: {name}")
    with _lock:
        _selected["embed"] = name


def embed_prefixes(name: str) -> tuple:
    """(query_prefix, document_prefix) for an embedding model."""
    info = EMBED_MODELS.get(name, {})
    return info.get("query_prefix", ""), info.get("doc_prefix", "")


def installed_models():
    """Set of installed Ollama model names, or None if Ollama is unreachable."""
    try:
        client = ollama.Client(host=OLLAMA_HOST, timeout=5)
        return {m.model for m in client.list().models}
    except Exception:
        return None


def is_installed(name: str, installed) -> bool:
    if not installed:
        return False
    return name in installed or (":" not in name and f"{name}:latest" in installed)

_DEFAULT_ANCHORS = {"skill_floor": 0.30, "skill_ceiling": 0.70, "chunk_floor": 0.30, "chunk_ceiling": 0.65}

def anchors(name: str) -> dict:
    """Cosine values that map to 0 and 100 points, per embedding model."""
    info = EMBED_MODELS.get(name, {})
    return {key: info.get(key, default) for key, default in _DEFAULT_ANCHORS.items()}