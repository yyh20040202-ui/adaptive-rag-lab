import pytest
from fastapi.testclient import TestClient
from adaptive_rag.data import Document
from adaptive_rag.engine import RetrievalEngine
from adaptive_rag.generator import local_base_url
from adaptive_rag.service import app, get_engine


class OfflineGenerator:
    def healthy(self):
        return False


@pytest.fixture
def client(monkeypatch):
    engine = RetrievalEngine([Document("d", "Hotel policy", "Hotel limit is CNY 500.")], generator=OfflineGenerator())
    monkeypatch.setattr("adaptive_rag.service.get_engine", lambda corpus: engine)
    return TestClient(app)


def test_offline_retrieval_still_runs_and_generation_abstains(client):
    data = {"query": "hotel limit", "mode": "bm25"}
    response = client.post("/api/search", json=data)
    assert response.status_code == 200
    assert response.json()["hits"][0]["source_id"] == "d"
    answer = client.post("/api/ask", json=data).json()
    assert answer["abstain"] and answer["audit_failures"] == ["local_generator_unavailable"]


def test_bounded_inputs_and_immutable_benchmark(client):
    assert client.post("/api/ask", json={"query": "hotel", "evidence_budget": 100000}).status_code == 422
    assert client.post("/api/documents", json={"title": "x", "text": "x"*20001}).status_code == 422
    assert client.post("/api/documents/pdf", json={"title": "x", "content_base64": "not-a-pdf"}).status_code == 400
    engine = RetrievalEngine([Document("d", "Hotel", "hotel price")], domain="scifact")
    with pytest.raises(ValueError):
        engine.add_document(Document("d2", "More", "more hotel"))


def test_user_corpus_starts_empty_without_simulated_data():
    engine = RetrievalEngine.from_corpus("demo")
    assert engine.documents == []
    assert engine.search("hotel allowance")["hits"] == []


def test_remote_generation_endpoints_are_rejected():
    assert local_base_url("http://127.0.0.1:8087/") == "http://127.0.0.1:8087"
    for endpoint in ["https://example.com", "http://example.com", "http://127.0.0.1@evil.example", "file:///etc/passwd"]:
        with pytest.raises(ValueError):
            local_base_url(endpoint)


def test_imported_document_enters_search_index(client):
    response = client.post("/api/documents", json={"title": "Unique topic", "text": "The quokka allowance is CNY 321."})
    assert response.status_code == 200
    result = client.post("/api/search", json={"query": "quokka", "mode": "bm25"}).json()
    assert result["hits"][0]["source_id"] == response.json()["id"]
