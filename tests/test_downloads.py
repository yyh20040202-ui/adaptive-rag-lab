import hashlib
import io
import pytest
from adaptive_rag.prepare import fetch


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


def test_interrupted_large_download_resumes_and_verifies_full_hash(tmp_path, monkeypatch):
    body = b"GGUF" + bytes(range(256))*4
    requests = []
    def open_url(request, **kwargs):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return Response(body[:301], headers={"Content-Length": str(len(body))})
        assert request.get_header("Range") == "bytes=301-"
        return Response(body[301:], 206, {"Content-Range": f"bytes 301-{len(body)-1}/{len(body)}"})
    monkeypatch.setattr("adaptive_rag.prepare.urllib.request.urlopen", open_url)
    monkeypatch.setattr("adaptive_rag.prepare.time.sleep", lambda _: None)
    path = tmp_path/"model.gguf"
    result = fetch("https://example.test/model", path, hashlib.sha256(body).hexdigest(), len(body))
    assert requests == [None, "bytes=301-"]
    assert path.read_bytes() == body and result["bytes"] == len(body)


def test_server_ignoring_range_restarts_without_appending(tmp_path, monkeypatch):
    body = b"verified model bytes"
    path = tmp_path/"model.gguf"
    path.with_suffix(".gguf.part").write_bytes(b"old incomplete data")
    monkeypatch.setattr("adaptive_rag.prepare.urllib.request.urlopen", lambda request, **kwargs: Response(body, headers={"Content-Length": str(len(body))}))
    result = fetch("https://example.test/model", path, hashlib.sha256(body).hexdigest(), len(body))
    assert path.read_bytes() == body and result["bytes"] == len(body)


def test_corrupt_download_never_becomes_ready_model(tmp_path, monkeypatch):
    path = tmp_path/"model.gguf"
    monkeypatch.setattr("adaptive_rag.prepare.urllib.request.urlopen", lambda request, **kwargs: Response(b"wrong", headers={"Content-Length": "5"}))
    monkeypatch.setattr("adaptive_rag.prepare.time.sleep", lambda _: None)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch("https://example.test/model", path, "0"*64, 5)
    assert not path.exists()
