import numpy as np
import httpx

from adaptive_rag.cross_encoder import blend_order
from adaptive_rag.data import Document
from adaptive_rag.engine import RetrievalEngine
from adaptive_rag.retrieval import normalized_scores, reciprocal_rank_fusion, top_indices


class FixedEncoder:
    def encode(self, texts):
        return np.tile(np.array([.1, .2, .3, .4], dtype=np.float32), (len(texts), 1))


class FixedCrossEncoder:
    def __init__(self):
        self.calls = []

    def predict(self, query, texts):
        self.calls.append((query, texts))
        return np.array([float("DCBA".index(text[0])) for text in texts], dtype=np.float32)


def offline_engine(monkeypatch):
    documents = [Document(letter, letter, "target " * (index+1)) for index, letter in enumerate("ABCD")]
    engine = RetrievalEngine(documents, domain="scifact", embeddings=np.eye(4, dtype=np.float32))
    engine.encoder = FixedEncoder()
    engine._dense_ready = lambda: None
    monkeypatch.setattr("adaptive_rag.engine.check_resources", lambda: {})
    return engine


def test_blend_uses_candidates_as_global_indices_and_returns_aligned_scores():
    candidates = np.array([4, 1, 6, 0])
    logits = np.array([-2., 3., 1., -1.])
    fusion = np.array([.2, .6, 0., 0., .8, 0., .4])
    order, values = blend_order(candidates, logits, fusion, alpha=.75, depth=3)
    expected = .75*normalized_scores(logits[:3]) + .25*normalized_scores(fusion[candidates[:3]])
    np.testing.assert_allclose(values, expected)
    assert order.tolist() == candidates[:3][top_indices(expected, 3)].tolist()
    zero_order, _ = blend_order(candidates, logits, fusion, alpha=0., depth=3)
    assert zero_order.tolist() == [4, 1, 6]
    ce_order, _ = blend_order(candidates, logits, fusion, alpha=1., depth=3)
    assert ce_order.tolist() == [1, 6, 4]


def test_hybrid_inference_honors_saved_candidate_depth_and_rrf_constant(monkeypatch):
    engine = offline_engine(monkeypatch)
    engine.config = {"candidate_depth": 2, "rrf_constant": 5}
    calls = []

    def fusion(rankings, count, constant=60, depth=100):
        calls.append((constant, depth, [len(ranking) for ranking in rankings]))
        return reciprocal_rank_fusion(rankings, count, constant, depth)

    monkeypatch.setattr("adaptive_rag.engine.reciprocal_rank_fusion", fusion)
    result = engine.search("target", "hybrid_lora", top_k=4)
    assert calls == [(5, 2, [2, 2])]
    assert len(result["hits"]) == 2


def test_served_crossencoder_blend_matches_offline_configuration(monkeypatch):
    engine = offline_engine(monkeypatch)
    engine.config = {"candidate_depth": 1, "rrf_constant": 60}
    engine.quality_config = {"candidate_depth": 3, "rrf_constant": 7,
                             "rerank_depth": 2, "alpha": .75,
                             "model_path": "unused", "max_length": 384}
    engine.cross_encoder = FixedCrossEncoder()
    query = "target"
    sparse = engine.bm25.scores(query)
    dense = engine.embeddings @ engine.encoder.encode([query])[0]
    fused = reciprocal_rank_fusion([top_indices(sparse, 3), top_indices(dense, 3)], 4, 7, 3)
    candidates = top_indices(fused, 3)
    texts = [engine.units[i].content for i in candidates[:2]]
    logits = engine.cross_encoder.predict(query, texts)
    order, _ = blend_order(candidates, logits, fused, .75, 2)
    engine.cross_encoder.calls.clear()
    actual = engine.search(query, "cross_encoder", top_k=2)
    assert engine.cross_encoder.calls == [(query, texts)]
    assert [item["source_id"] for item in actual["hits"]] == [engine.documents[i].id for i in order]


def test_long_local_document_retrieves_late_chunk_with_original_offsets():
    source = Document("source", "Travel policy", "Routine filing instructions. " * 180 + "Quokka reimbursement is CNY 321.")
    other = Document("other", "Unrelated policy", "Office door instructions.")
    engine = RetrievalEngine([source, other], domain="demo")
    result = engine.search("quokka", "bm25", top_k=2)
    hit = result["hits"][0]
    assert hit["source_id"] == source.id
    span = hit["retrieved_span"]
    assert span["start"] > 1200
    assert span["text"] == source.content[span["start"]:span["end"]]
    assert "Quokka reimbursement is CNY 321." in span["text"]
    assert len({item["source_id"] for item in result["hits"]}) == len(result["hits"])


def test_auto_falls_back_when_downloaded_gpu_reranker_is_offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path/"reranker.gguf"
    weights.write_bytes(b"fixture")
    engine = RetrievalEngine([Document("doc", "Study", "Memory cells control infection.")])
    engine.quality_config = {"model_path": str(tmp_path), "model_file": weights.name,
                             "backend": "llama_vulkan"}

    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("owned server was stopped")

    monkeypatch.setattr("adaptive_rag.engine.httpx.get", unavailable)
    result = engine.search("memory cells", "auto")
    assert result["route"] == "bm25"
    assert result["hits"][0]["source_id"] == "doc"
