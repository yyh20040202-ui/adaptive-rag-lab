"""Content-addressed embedding caches; IDs alone cannot prove text consistency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data import Document, file_sha256

CACHE_SCHEMA = 2
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _digest_rows(rows) -> str:
    """JSON delimiters preserve Unicode, newlines and unambiguous row boundaries."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_hashes(directory: str | Path, names: Sequence[str]) -> dict[str, str]:
    directory = Path(directory)
    return {name: file_sha256(directory / name) for name in names}


def make_embedding_fingerprint(documents: Sequence[Document], queries: Mapping[str, str],
                               query_ids: Sequence[str] | None = None, *,
                               model_path: str | Path = "models/minilm",
                               model_id: str | None = None, model_revision: str | None = None,
                               adapter_path: str | Path | None = None, max_length: int = 192,
                               dimension: int = 384) -> dict:
    """Describe exactly the inputs and inference settings that produced vectors.

    The model/tokenizer and adapter hashes describe actual local bytes, in
    addition to their declared public revision. No absolute author paths appear
    in the fingerprint. Query order is explicit because embeddings are arrays.
    """
    if model_id is None or model_revision is None:
        from .prepare import ENCODER_ID, ENCODER_REVISION
        model_id = model_id or ENCODER_ID
        model_revision = model_revision or ENCODER_REVISION
    ids = [document.id for document in documents]
    qids = list(sorted(queries) if query_ids is None else query_ids)
    if len(ids) != len(set(ids)) or len(qids) != len(set(qids)):
        raise ValueError("Embedding inputs require unique document/query IDs")
    if not set(qids) <= queries.keys():
        raise ValueError("Missing query text for embedding cache")
    if max_length <= 0 or dimension <= 0:
        raise ValueError("Embedding length and dimension must be positive")
    result = {
        "schema": CACHE_SCHEMA,
        "documents": {"count": len(ids), "ids_sha256": _digest_rows(ids),
                      "texts_sha256": _digest_rows((d.id, d.content) for d in documents)},
        "queries": {"count": len(qids), "ids_sha256": _digest_rows(qids),
                    "texts_sha256": _digest_rows((qid, queries[qid]) for qid in qids)},
        "encoder": {"model_id": model_id, "revision": model_revision,
                    "files_sha256": _file_hashes(model_path, MODEL_FILES),
                    "max_length": max_length, "dimension": dimension,
                    "pooling": "attention-mask-mean-l2-v1", "dtype": "float32"},
        "adapter": None if adapter_path is None else _file_hashes(adapter_path, ADAPTER_FILES),
    }
    result["fingerprint_sha256"] = hashlib.sha256(_json(result).encode("utf-8")).hexdigest()
    return result


def _validate_arrays(documents: np.ndarray, queries: np.ndarray, document_ids: Sequence[str],
                     query_ids: Sequence[str], fingerprint: dict) -> None:
    dimension = fingerprint["encoder"]["dimension"]
    for name, values, ids in (("documents", documents, document_ids), ("queries", queries, query_ids)):
        expected = fingerprint[name]
        if values.shape != (expected["count"], dimension):
            raise ValueError(f"Embedding cache {name} shape mismatch")
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"Embedding cache {name} must contain finite float32 vectors")
        if len(ids) != expected["count"] or _digest_rows(list(ids)) != expected["ids_sha256"]:
            raise ValueError(f"Embedding cache {name} IDs/order mismatch")


def load_embedding_cache(path: str | Path, expected: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Return validated matrices, None for absence, or fail on stale/corrupt data.

    Legacy caches lack text/model provenance and are deliberately not migrated
    by adding metadata to old vectors. A one-time reindex is required.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if "fingerprint_json" not in arrays:
                raise ValueError("legacy cache has no text/model fingerprint; rebuild once")
            actual = json.loads(str(arrays["fingerprint_json"].item()))
            if actual != expected:
                raise ValueError("text, IDs, model, adapter or encoder settings changed; rebuild")
            documents, queries = arrays["documents"], arrays["queries"]
            _validate_arrays(documents, queries, arrays["document_ids"].tolist(),
                             arrays["query_ids"].tolist(), expected)
            return documents, queries
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid embedding cache {path.name}: {error}") from error


def save_embedding_cache(path: str | Path, documents: np.ndarray, queries: np.ndarray,
                         document_ids: Sequence[str], query_ids: Sequence[str], fingerprint: dict) -> None:
    """Validate and atomically replace a cache; keep old field names readable."""
    _validate_arrays(documents, queries, document_ids, query_ids, fingerprint)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    extra = {}
    if fingerprint["adapter"] is not None:
        extra["adapter_sha256"] = fingerprint["adapter"]["adapter_model.safetensors"]
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, documents=documents, queries=queries,
                            document_ids=np.asarray(document_ids, dtype=str),
                            query_ids=np.asarray(query_ids, dtype=str),
                            fingerprint_json=np.asarray(_json(fingerprint)), **extra)
    temporary.replace(path)
