"""Pinned public model/data downloads. Never executes code from model repositories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import ssl
import time
import urllib.request
import zipfile
import certifi
from .data import file_sha256

ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
GENERATOR_ID = "ggml-org/Qwen3-0.6B-GGUF"
GENERATOR_REVISION = "b5f37287796e5be0ea3dab2e7430873fb3f73e49"
DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11517/llama-b11517-bin-win-vulkan-x64.zip"
LLAMA_SHA256 = "bd2ff142b9f2df12051b78ce0b68bf8cdf11c8fd897c6748c5823320bcb01b60"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def get_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "adaptive-rag-lab/0.1"})
    with urllib.request.urlopen(request, timeout=40, context=TLS_CONTEXT) as response:
        return json.load(response)


def fetch(url: str, path: Path, expected_sha: str | None = None, expected_size: int | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size) and (expected_sha is None or file_sha256(path) == expected_sha):
        return {"path": path.name, "url": url, "bytes": path.stat().st_size,
                "sha256": file_sha256(path)}
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if expected_size is not None and offset >= expected_size:
                if offset == expected_size and (expected_sha is None or file_sha256(temporary) == expected_sha):
                    temporary.replace(path)
                    return {"path": path.name, "url": url, "bytes": expected_size, "sha256": file_sha256(path)}
                offset = 0
            headers = {"User-Agent": "adaptive-rag-lab/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60, context=TLS_CONTEXT) as response:
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if match is None or int(match[1]) != offset:
                        raise ValueError("Invalid resumed response range")
                    total_size = int(match[3])
                    if expected_size is not None and total_size != expected_size:
                        raise ValueError("Remote size changed from pinned metadata")
                else:
                    offset = 0  # server declined Range; restart rather than append
                    total_size = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else expected_size
                with temporary.open("ab" if offset else "wb") as handle:
                    count, report_at = offset, offset+32*1024*1024
                    while chunk := response.read(2*1024*1024):
                        handle.write(chunk)
                        count += len(chunk)
                        if count >= report_at:
                            print(f"{path.name}: {count//(1024*1024)} MiB", flush=True)
                            report_at += 32*1024*1024
            required_size = expected_size if expected_size is not None else total_size
            if required_size is not None and count != required_size:
                raise OSError(f"Download truncated at {count}/{required_size} bytes; next attempt resumes")
            digest = file_sha256(temporary)
            if expected_sha and digest != expected_sha:
                temporary.write_bytes(b"")  # fully downloaded corrupt file must not be resumed
                raise ValueError(f"Checksum mismatch for {path.name}")
            temporary.replace(path)
            return {"path": path.name, "url": url, "bytes": count, "sha256": digest}
        except Exception as error:
            if attempt == 2:
                raise
            print(f"Retry {attempt+1}/3 for {path.name}: {type(error).__name__}: {error}", flush=True)
            time.sleep(2)
    raise RuntimeError("Unreachable download state")


def download_encoder(destination: Path) -> list:
    tree = get_json(f"https://huggingface.co/api/models/{ENCODER_ID}/tree/{ENCODER_REVISION}")
    hashes = {v["path"]: v.get("lfs", {}).get("oid") for v in tree}
    files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "vocab.txt", "model.safetensors"]
    records = [fetch(f"https://huggingface.co/{ENCODER_ID}/resolve/{ENCODER_REVISION}/{name}?download=true",
                     destination / name, hashes.get(name)) for name in files]
    (destination / "provenance.json").write_text(json.dumps({"model": ENCODER_ID,
                "revision": ENCODER_REVISION, "license": "Apache-2.0", "files": records}, indent=2))
    return records


def download_data(destination: Path) -> dict:
    archive = destination / "scifact.zip"
    metadata = fetch(DATA_URL, archive, DATA_SHA256)
    allowed = {"corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv"}
    with zipfile.ZipFile(archive) as handle:
        for name in allowed:
            candidates = [member for member in handle.namelist()
                          if member == name or member.endswith("/" + name)]
            if len(candidates) != 1:
                raise ValueError(f"Archive must contain exactly one {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(handle.read(candidates[0]))
    (destination / "provenance.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def download_generator(destination: Path) -> dict:
    name = "Qwen3-0.6B-Q4_0.gguf"
    tree = get_json(f"https://huggingface.co/api/models/{GENERATOR_ID}/tree/{GENERATOR_REVISION}")
    match = next(v for v in tree if v["path"] == name)
    result = fetch(f"https://huggingface.co/{GENERATOR_ID}/resolve/{GENERATOR_REVISION}/{name}?download=true",
                   destination / name, match.get("lfs", {}).get("oid"))
    (destination / "provenance.json").write_text(json.dumps({"model": GENERATOR_ID,
                 "base_model": "Qwen/Qwen3-0.6B", "revision": GENERATOR_REVISION,
                 "license": "Apache-2.0", "file": result}, indent=2))
    return result


def download_llama_windows(destination: Path) -> dict:
    archive = destination / "llama.zip"
    metadata = fetch(LLAMA_URL, archive, LLAMA_SHA256)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive member")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                content = handle.read(member)
                if target.exists() and hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(content).digest():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    (destination / "provenance.json").write_text(json.dumps({"version": "b11517", "backend": "Vulkan", **metadata}, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", action="store_true", help="Also download Qwen GGUF")
    parser.add_argument("--llama-win", action="store_true", help="Also download pinned Windows Vulkan llama.cpp")
    args = parser.parse_args()
    print(json.dumps(download_data(Path("data/raw/scifact")), indent=2), flush=True)
    print(json.dumps(download_encoder(Path("models/minilm")), indent=2), flush=True)
    if args.generator:
        print(json.dumps(download_generator(Path("models/qwen")), indent=2), flush=True)
    if args.llama_win:
        print(json.dumps(download_llama_windows(Path("models/llama")), indent=2), flush=True)
