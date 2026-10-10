"""Conditional retrieval shared by the UI, API and executable examples."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import numpy as np
import httpx

from .data import Document, file_sha256, load_benchmark
from .answerability import has_topical_overlap, missing_or_echo_answer, missing_query_anchor
from .evidence import abstention, pack_evidence, validate_answer
from .generator import LocalGenerator
from .resources import HEAVY_JOB_LOCK, check_resources
from .retrieval import (BM25, normalized_scores, pair_features, reciprocal_rank_fusion,
                        routing_features, tokenize, top_indices)


class RetrievalEngine:
    def __init__(self, documents: list[Document], domain: str = "demo",
                 embeddings: np.ndarray | None = None, generator=None):
        if len({d.id for d in documents}) != len(documents):
            raise ValueError("Documents must have unique IDs")
        self.documents = documents
        self.domain = domain
        self._rebuild_units()
        self.embeddings = embeddings
        self.encoder = self.ranker = self.router = self.cross_encoder = None
        self.lock = threading.RLock()
        self.config = {}
        config_path = Path("checkpoints/config.json")
        if config_path.exists():
            self.config = json.loads(config_path.read_text(encoding="utf-8"))
        quality_path = Path("checkpoints/quality_config.json")
        self.quality_config = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
        self.generator = generator or LocalGenerator()

    def _rebuild_units(self):
        if self.domain == "demo":
            from .chunking import chunk_document
            chunks = [chunk for document in self.documents for chunk in chunk_document(document)]
            self.units = [Document(c.id, "", c.text) for c in chunks]
            self.unit_sources = {c.id: (c.source_id, c.start, c.end) for c in chunks}
        else:
            self.units = self.documents
            self.unit_sources = {d.id: (d.id, 0, len(d.content)) for d in self.documents}
        self.bm25 = BM25([d.content for d in self.units]) if self.units else None

    @classmethod
    def from_corpus(cls, domain: str):
        if domain == "scifact":
            benchmark = load_benchmark()
            from .cache import make_embedding_fingerprint, load_embedding_cache
            cache = Path("artifacts/lora_embeddings.npz")
            arrays = None
            if cache.exists():
                fingerprint = make_embedding_fingerprint(benchmark.documents, benchmark.queries,
                                   adapter_path="checkpoints/retriever_lora", max_length=192)
                arrays = load_embedding_cache(cache, fingerprint)
            return cls(benchmark.documents, domain, arrays[0] if arrays is not None else None)
        if domain != "demo":
            raise ValueError("Unknown corpus")
        return cls([], domain)  # local user documents start empty; no simulated business data

    def _dense_ready(self):
        if not Path("models/minilm/model.safetensors").exists():
            raise FileNotFoundError("MiniLM not downloaded. Run python -m adaptive_rag.prepare")
        if self.encoder is None:
            from .encoder import Encoder
            adapter = "checkpoints/retriever_lora" if Path("checkpoints/retriever_lora/adapter_model.safetensors").exists() else None
            self.encoder = Encoder(adapter=adapter, max_length=self.config.get("encoder_max_length", 192))
        if self.embeddings is None:
            self.embeddings = self.encoder.encode([d.content for d in self.units])

    def _heads_ready(self):
        from .ranking import Ranker, Router, load_head
        if self.ranker is None:
            self.ranker = load_head(Ranker(), "checkpoints/ranker.safetensors")
        if self.router is None:
            self.router = load_head(Router(), "checkpoints/router.safetensors")

    def _quality_ready(self) -> bool:
        cfg = self.quality_config
        if not cfg or not (Path(cfg["model_path"])/cfg.get("model_file", "model.safetensors")).exists():
            return False
        if cfg.get("backend") != "llama_vulkan":
            return True
        from .generator import local_base_url
        try:
            return httpx.get(local_base_url(os.getenv("RERANK_BASE_URL", "http://127.0.0.1:8089"))+"/health",
                             timeout=1, trust_env=False).status_code == 200
        except httpx.HTTPError:
            return False

    def search(self, query: str, mode: str = "auto", top_k: int = 5) -> dict:
        if mode == "auto":
            if self._quality_ready():
                mode = "cross_encoder"
            elif Path("models/minilm/model.safetensors").exists() and Path("checkpoints/ranker.safetensors").exists():
                mode = "neural_rerank"
            else:
                mode = "bm25"
        if mode not in {"cross_encoder", "adaptive", "bm25", "hybrid_lora", "neural_rerank"}:
            raise ValueError("Unknown retrieval mode")
        if not query.strip() or len(query) > 1000 or not 1 <= top_k <= 20:
            raise ValueError("Invalid query or top_k")
        with HEAVY_JOB_LOCK, self.lock:
            started = time.perf_counter()
            if not self.documents:
                return {"query": query, "domain": self.domain, "route": "bm25", "policy": "empty_local_corpus",
                        "hits": [], "trace": [], "retrieval_ms": (time.perf_counter()-started)*1000}
            sparse = self.bm25.scores(query)
            depth = self.config.get("candidate_depth", 100)
            constant = self.config.get("rrf_constant", 60)
            if mode == "cross_encoder":
                if not self.quality_config:
                    raise FileNotFoundError("Run adaptive_rag.quality --stage select before quality mode")
                depth = self.quality_config["candidate_depth"]
                constant = self.quality_config["rrf_constant"]
            br = top_indices(sparse, depth)
            route, policy = mode, "explicit"
            if mode == "adaptive":
                if self.domain == "scifact" and self.config:
                    self._heads_ready()
                    routes = self.config["routes"]
                    route = routes[int(self.router.predict(routing_features(query, sparse, self.bm25))[0])]
                    policy = "learned_scifact_router"
                else:
                    # Training is English biomedical: no unmeasured generalization claim.
                    route, policy = "bm25", "local_conservative_policy"
            trace = [{"step": "BM25", "executed": True},
                     {"step": "MiniLM LoRA", "executed": route != "bm25"},
                     {"step": "Neural reranker", "executed": route == "neural_rerank"},
                     {"step": "Token cross-encoder", "executed": route == "cross_encoder"}]
            order, scores = br, sparse
            if route != "bm25":
                check_resources()
                self._dense_ready()
                query_vector = self.encoder.encode([query])[0]
                dense = self.embeddings @ query_vector
                dr = top_indices(dense, depth)
                fused = reciprocal_rank_fusion([br, dr], len(self.units), constant, depth)
                order, scores = top_indices(fused, depth), fused
                if route == "neural_rerank":
                    self._heads_ready()
                    terms = set(tokenize(query))
                    coverage = np.array([len(terms & self.bm25.text_tokens[i])/max(len(terms), 1) for i in order])
                    predicted = self.ranker.predict(pair_features(query_vector, self.embeddings[order],
                                                    sparse[order], dense[order], coverage))
                    alpha = self.config.get("rerank_alpha", 0)
                    blended = (1-alpha)*normalized_scores(fused[order]) + alpha*predicted
                    scores = np.zeros(len(self.units), dtype=np.float32)
                    scores[order] = blended
                    order = order[top_indices(blended, len(order))]
                elif route == "cross_encoder":
                    from .cross_encoder import blend_order
                    from .reranker_client import create_reranker
                    cfg = self.quality_config
                    if self.cross_encoder is None:
                        self.cross_encoder = create_reranker(cfg)
                    candidates = order[:cfg["rerank_depth"]]
                    logits = self.cross_encoder.predict(query, [self.units[i].content for i in candidates])
                    order, values = blend_order(candidates, logits, fused, cfg["alpha"], len(candidates))
                    scores = np.zeros(len(self.units), dtype=np.float32)
                    scores[candidates] = values
            originals = {d.id: d for d in self.documents}
            hits, seen = [], set()
            for i in order:
                source_id, start, end = self.unit_sources[self.units[i].id]
                if source_id in seen:
                    continue
                seen.add(source_id)
                doc = originals[source_id]
                hits.append({"source_id": source_id, "title": doc.title, "text": doc.content,
                             "score": float(scores[i]), "bm25_score": float(sparse[i]),
                             "retrieved_span": {"start": start, "end": end, "text": doc.content[start:end]}})
                if len(hits) >= top_k:
                    break
            return {"query": query, "domain": self.domain, "route": route, "policy": policy,
                    "hits": hits, "trace": trace, "retrieval_ms": (time.perf_counter()-started)*1000}

    def ask(self, query: str, mode: str = "auto", evidence_budget: int = 1400) -> dict:
        # Serialize the whole pipeline, including GPU generation, across corpora.
        # search() takes this reentrant lock again; status requests remain responsive.
        with HEAVY_JOB_LOCK, self.lock:
            return self._ask(query, mode, evidence_budget)

    def _ask(self, query: str, mode: str, evidence_budget: int) -> dict:
        if not 128 <= evidence_budget <= 1800:
            raise ValueError("Evidence budget must lie in [128, 1800]")
        result = self.search(query, mode, top_k=5)
        trace = result.copy()
        trace.pop("hits", None)
        if not result["hits"] or max(h["bm25_score"] for h in result["hits"]) <= 0:
            return {**abstention(["no_lexical_evidence"]), **trace, "evidence": [], "evidence_tokens": 0}
        if not has_topical_overlap(query, [h["text"] for h in result["hits"]]):
            return {**abstention(["no_topical_overlap"]), **trace, "evidence": result["hits"], "evidence_tokens": 0}
        if not self.generator.healthy():
            return {**abstention(["local_generator_unavailable"]), **trace, "evidence": result["hits"],
                    "evidence_tokens": None}
        docs_by_id = {d.id: d for d in self.documents}
        selected = [docs_by_id[h["source_id"]] for h in result["hits"]]
        if hasattr(self.generator, "last_attempt"):
            self.generator.last_attempt = {}
        try:
            preferred = {hit["source_id"]: hit["retrieved_span"] for hit in result["hits"]} if self.domain == "demo" else None
            evidence, tokens = pack_evidence(selected, query, evidence_budget, self.generator.count_tokens, preferred_spans=preferred)
            if not evidence:
                return {**abstention(["budget_has_no_evidence"]), **trace, "evidence": [], "evidence_tokens": tokens}
            if self.domain == "scifact" and missing_query_anchor(query, [item.text for item in evidence]):
                from dataclasses import asdict
                return {**abstention(["missing_query_anchor"]), **trace,
                        "evidence": [asdict(item) for item in evidence], "evidence_tokens": tokens,
                        "evidence_budget": evidence_budget}
            payload, stats = self.generator.generate(query, evidence)
            answer = validate_answer(payload, evidence)
            if not answer["abstain"]:
                reason = missing_or_echo_answer(query, answer["answer"])
                if reason:
                    answer = abstention([reason])
            from dataclasses import asdict
            return {**answer, **trace, **stats, "evidence": [asdict(e) for e in evidence],
                    "evidence_tokens": tokens, "evidence_budget": evidence_budget,
                    "audit_scope": "source IDs and exact quote offsets; semantic entailment is not certified"}
        except (ValueError, KeyError, TypeError, httpx.HTTPError) as error:
            attempt = getattr(self.generator, "last_attempt", {})
            return {**abstention(["generation_or_format_error"]), **trace,
                    **attempt, "generation_error_kind": type(error).__name__,
                    "evidence": [], "evidence_tokens": None}

    def add_document(self, document: Document) -> None:
        with self.lock:
            if self.domain != "demo":
                raise ValueError("Benchmark corpus is immutable")
            if document.id in {d.id for d in self.documents}:
                raise ValueError("Duplicate document ID")
            if len(self.documents) >= 40:
                raise ValueError("Demo document limit reached")
            self.documents.append(document)
            self._rebuild_units()
            self.embeddings = None
