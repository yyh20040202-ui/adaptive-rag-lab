"""Rebuild an index with the committed adapter; no training required."""
from pathlib import Path
import numpy as np
from .data import load_benchmark
from .cache import make_embedding_fingerprint, save_embedding_cache
from .encoder import Encoder


if __name__ == "__main__":
    data = load_benchmark()
    encoder = Encoder(adapter="checkpoints/retriever_lora", max_length=192)
    ids = [d.id for d in data.documents]
    qids = sorted(data.queries)
    documents = encoder.encode([d.content for d in data.documents], progress=True)
    queries = encoder.encode([data.queries[qid] for qid in qids], progress=True)
    Path("artifacts").mkdir(exist_ok=True)
    fingerprint = make_embedding_fingerprint(data.documents, data.queries, qids,
                                              adapter_path="checkpoints/retriever_lora", max_length=192,
                                              dimension=encoder.model.config.hidden_size)
    save_embedding_cache("artifacts/lora_embeddings.npz", documents, queries, ids, qids, fingerprint)
    print(f"Indexed {len(ids)} documents and {len(qids)} queries", flush=True)
