"""Exercise real local API data and static assets, without a simulated LLM."""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from .data import load_benchmark
from .service import app, get_engine


if __name__ == "__main__":
    client = TestClient(app)
    checks = {}
    for route in ["/", "/static/app.js", "/static/style.css", "/api/health", "/api/resources", "/api/reports", "/api/example-queries", "/api/documents?corpus=scifact"]:
        response = client.get(route)
        checks[route] = response.status_code == 200
    data = load_benchmark()
    qid = sorted(data.test_qrels)[0]
    response = client.post("/api/search", json={"query": data.queries[qid], "corpus": "scifact", "mode": "bm25", "top_k": 10})
    checks["real_benchmark_search"] = response.status_code == 200 and len(response.json()["hits"]) == 10
    status = client.get("/api/health").json()
    if status.get("cross_encoder_available"):
        response = client.post("/api/search", json={"query": data.queries[next(iter(data.train_qrels))],
                                                    "corpus": "scifact", "mode": "cross_encoder", "top_k": 5})
        checks["selected_quality_model_search"] = response.status_code == 200 and response.json()["route"] == "cross_encoder" and len(response.json()["hits"]) == 5
    document = data.documents[0]
    imported = client.post("/api/documents", json={"title": document.title, "text": document.text})
    checks["real_paper_import"] = imported.status_code == 200
    found = client.post("/api/search", json={"query": document.title, "corpus": "demo", "mode": "bm25"})
    checks["imported_paper_search"] = found.status_code == 200 and found.json()["hits"][0]["source_id"] == imported.json()["id"]
    checks["immutable_benchmark"] = client.post("/api/documents", json={"title": "x", "text": "x"*20001}).status_code == 422
    report = {"passed": all(checks.values()), "checks": checks,
              "scope": "In-process actual FastAPI endpoints; real SciFact text, no LLM mock. Static assets served; browser layout/interaction not visually verified."}
    Path("reports/api_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    get_engine.cache_clear()
    if not report["passed"]:
        raise SystemExit(1)
