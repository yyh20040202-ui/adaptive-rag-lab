"""Local FastAPI workbench with bounded document input and no cloud calls."""
from __future__ import annotations

import base64
import binascii
from functools import lru_cache
import io
import json
from pathlib import Path
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import Document
from .engine import RetrievalEngine
from .generator import LocalGenerator
from .resources import ResourcePressure, snapshot

app = FastAPI(title="Adaptive RAG Lab", version="0.1.0")
web = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=web), name="static")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    corpus: str = Field(default="scifact", pattern="^(demo|scifact)$")
    mode: str = Field(default="auto", pattern="^(auto|cross_encoder|adaptive|bm25|hybrid_lora|neural_rerank)$")
    evidence_budget: int = Field(default=1400, ge=128, le=1800)
    top_k: int = Field(default=5, ge=1, le=20)


class DocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    text: str = Field(min_length=1, max_length=20000)


class PDFRequest(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    content_base64: str = Field(min_length=1, max_length=2800000)


@lru_cache(maxsize=2)
def get_engine(corpus: str) -> RetrievalEngine:
    return RetrievalEngine.from_corpus(corpus)


@app.get("/")
def index():
    return FileResponse(web / "index.html")


@app.get("/api/health")
def health():
    quality_path = Path("checkpoints/quality_config.json")
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    encoder_ready = Path("models/minilm/model.safetensors").exists()
    ce_ready = bool(encoder_ready and quality and (Path(quality["model_path"])/quality.get("model_file", "model.safetensors")).exists())
    if ce_ready and quality.get("backend") == "llama_vulkan":
        import httpx
        from .generator import local_base_url
        import os
        try:
            ce_ready = httpx.get(local_base_url(os.getenv("RERANK_BASE_URL", "http://127.0.0.1:8089"))+"/health", timeout=1, trust_env=False).status_code == 200
        except httpx.HTTPError:
            ce_ready = False
    modes = ["bm25"]
    if encoder_ready:
        modes = ["neural_rerank", "hybrid_lora", "adaptive", "bm25"]
    if ce_ready:
        modes.insert(0, "cross_encoder")
    return {"status": "ok", "generator_available": LocalGenerator().healthy(),
            "encoder_available": encoder_ready,
            "benchmark_available": Path("data/raw/scifact/corpus.jsonl").exists(),
            "adapter_available": Path("checkpoints/retriever_lora/adapter_model.safetensors").exists(),
            "model": LocalGenerator().model_label, "local_only": True,
            "cross_encoder_available": ce_ready,
            "retrieval_modes": modes, "default_mode": modes[0]}


@app.get("/api/resources")
def resources():
    return snapshot()


@app.get("/api/reports")
def reports():
    result = {}
    for name in ["training", "retrieval", "generation"]:
        path = Path("reports") / f"{name}.json"
        result[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    for key, filename in [("quality", "quality.json"), ("generation_real", "real_corpus_generation_v2.json")]:
        path = Path("reports") / filename
        result[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return result


def run_request(request: SearchRequest, answer: bool):
    try:
        engine = get_engine(request.corpus)
        if answer:
            return engine.ask(request.query, request.mode, request.evidence_budget)
        return engine.search(request.query, request.mode, request.top_k)
    except FileNotFoundError as error:
        raise HTTPException(503, "模型或数据尚未准备好，请按 README 下载并建立索引。") from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except ResourcePressure as error:
        raise HTTPException(503, str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(503, "本地重排服务未能完成请求，请检查运行日志和内存余量。") from error


@app.post("/api/search")
def search(request: SearchRequest):
    return run_request(request, False)


@app.post("/api/ask")
def ask(request: SearchRequest):
    return run_request(request, True)


@app.get("/api/documents")
def documents(corpus: str = "demo"):
    if corpus not in {"demo", "scifact"}:
        raise HTTPException(400, "Unknown corpus")
    try:
        engine = get_engine(corpus)
        return {"count": len(engine.documents), "documents": [{"id": d.id, "title": d.title,
                "characters": len(d.content), "preview": d.text[:180]} for d in engine.documents[:40]],
                "domain": corpus, "synthetic": False}
    except FileNotFoundError as error:
        raise HTTPException(503, "Benchmark has not been downloaded") from error


@app.get("/api/example-queries")
def example_queries():
    from .data import load_benchmark
    try:
        data = load_benchmark()
        # Readable demo cases, including one missing-evidence refusal. These are
        # selected examples, not a representative or independent QA evaluation.
        keys = [qid for qid in ["1226", "141", "380", "1"] if qid in data.test_qrels]
        return {"queries": [{"id": qid, "query": data.queries[qid]} for qid in keys],
                "scope": "Selected, previously inspected real claims for demonstration; not QA accuracy."}
    except FileNotFoundError:
        return {"queries": []}


def import_text(title: str, text: str):
    engine = get_engine("demo")
    # No supplied file paths are used. Imported content stays in this process's memory.
    import uuid
    document = Document("local-" + uuid.uuid4().hex[:12], title.strip(), text.strip())
    if not document.title or not document.text:
        raise HTTPException(400, "Title and extracted text must not be empty")
    try:
        engine.add_document(document)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return {"id": document.id, "characters": len(document.content), "storage": "in memory until restart"}


@app.post("/api/documents")
def add_document(request: DocumentRequest):
    return import_text(request.title, request.text)


@app.post("/api/documents/pdf")
def add_pdf(request: PDFRequest):
    try:
        data = base64.b64decode(request.content_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise HTTPException(400, "Invalid base64 PDF") from error
    if len(data) > 2*1024*1024 or not data.startswith(b"%PDF-"):
        raise HTTPException(400, "Use a text-based PDF no larger than 2 MiB")
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted or len(reader.pages) > 30:
            raise ValueError("Encrypted or excessive-page PDF")
        chunks, length = [], 0
        for page in reader.pages:
            text = page.extract_text() or ""
            length += len(text)
            if length > 20000:
                raise ValueError("Extracted text exceeds 20,000 characters")
            chunks.append(text)
        text = "\n".join(chunks)
    except Exception as error:
        raise HTTPException(400, "无法读取该 PDF；请使用最多30页、20000字的非加密文字版 PDF。") from error
    return import_text(request.title, text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8088)
