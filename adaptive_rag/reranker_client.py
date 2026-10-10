"""Bounded loopback GGUF pair scoring through the pinned Vulkan runtime."""
import json
from pathlib import Path
import os
from functools import lru_cache
import numpy as np
import httpx
from .generator import local_base_url
from .resources import ResourceRecorder


class GGUFReranker:
    def __init__(self, model_path="models/bge_reranker", max_length=384, batch_size=8):
        self.path = Path(model_path)
        self.provenance = json.loads((self.path/"provenance.json").read_text(encoding="utf-8"))
        self.base_url = local_base_url(os.getenv("RERANK_BASE_URL", "http://127.0.0.1:8089"))
        self.client = httpx.Client(base_url=self.base_url, timeout=120, trust_env=False)
        self.max_length = max_length
        self.batch_size = min(8, max(1, batch_size))
        self.recorder = ResourceRecorder()
        self.recorder.sample()
        if not self.healthy():
            raise FileNotFoundError("Start python -m adaptive_rag.launch_generator --model-directory models/bge_reranker --reranker")
        props = self.client.get("/props").json()
        actual = Path(props.get("model_path", "")).name
        expected = self.provenance["file"]["path"]
        if actual != expected:
            raise ValueError("The loopback reranker model differs from the pinned experiment")

    def healthy(self):
        try:
            return self.client.get("/health", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    @lru_cache(maxsize=6000)
    def tokenize(self, text):
        response = self.client.post("/tokenize", json={"content": text, "add_special": False})
        response.raise_for_status()
        return tuple(response.json()["tokens"])

    @lru_cache(maxsize=6000)
    def document_prefix(self, text, limit):
        tokens = self.tokenize(text)
        if len(tokens) <= limit:
            return text
        response = self.client.post("/detokenize", json={"tokens": tokens[:limit]})
        response.raise_for_status()
        return response.json()["content"]

    def predict(self, query, texts):
        query_tokens = len(self.tokenize(query))
        if query_tokens > 128:
            raise ValueError("质量模式的问题请控制在128个模型token以内。")
        limit = self.max_length-query_tokens-4
        docs = [self.document_prefix(text, limit) for text in texts]
        values = []
        for start in range(0, len(docs), self.batch_size):
            self.recorder.sample()
            batch = docs[start:start+self.batch_size]
            response = self.client.post("/v1/rerank", json={"query": query, "documents": batch, "top_n": len(batch)})
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("results", payload.get("data"))
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise ValueError("Reranker returned an incomplete pair batch")
            indices = [row["index"] for row in rows]
            if sorted(indices) != list(range(len(batch))):
                raise ValueError("Reranker returned invalid document indices")
            scores = [None]*len(batch)
            for row in rows:
                scores[row["index"]] = float(row["relevance_score"])
            values.extend(scores)
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Reranker returned non-finite scores")
        return array


def create_reranker(config):
    if config.get("backend") == "llama_vulkan":
        return GGUFReranker(config["model_path"], config["max_length"], 8)
    from .cross_encoder import CrossEncoder
    return CrossEncoder(config["model_path"], config["max_length"], 8)
